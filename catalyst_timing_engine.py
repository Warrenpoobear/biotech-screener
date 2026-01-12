#!/usr/bin/env python3
"""
catalyst_timing_engine.py - Timing Engine for Catalyst Module

Implements:
- Expected readout calculation with enrollment-based forecasting
- Sponsor delay factors (historical slippage database)
- PDUFA date tracking
- Catalyst clustering detection (convexity windows)
- Confidence intervals for timing estimates

Design Philosophy:
- DETERMINISTIC: Reproducible outputs from same inputs
- STDLIB-ONLY: No external dependencies
- PIT-SAFE: All dates explicit
- GOVERNED: Auditable timing estimates

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import json


# =============================================================================
# TIMING CONSTANTS
# =============================================================================

# Average months from last patient enrolled to data readout by phase
READOUT_LEAD_TIMES: Dict[str, int] = {
    "P1": 6,
    "P1_2": 12,
    "P2": 18,
    "P2_3": 24,
    "P3": 30,
}

# PDUFA review timelines (months)
PDUFA_STANDARD_REVIEW_MONTHS = 10
PDUFA_PRIORITY_REVIEW_MONTHS = 6


# =============================================================================
# SPONSOR DELAY DATABASE
# =============================================================================

class SponsorReliability(str, Enum):
    """Sponsor execution reliability tier."""
    TIER_1 = "TIER_1"  # Top performers, minimal delays
    TIER_2 = "TIER_2"  # Average performers
    TIER_3 = "TIER_3"  # Below average, frequent delays
    UNKNOWN = "UNKNOWN"


# Historical sponsor delay factors (months of typical slippage)
# Format: {sponsor_name: (tier, avg_delay_months, std_dev_months)}
SPONSOR_DELAY_DATABASE: Dict[str, Tuple[SponsorReliability, Decimal, Decimal]] = {
    # Tier 1 - Major pharma with strong execution
    "PFIZER": (SponsorReliability.TIER_1, Decimal("1.5"), Decimal("1.0")),
    "MERCK": (SponsorReliability.TIER_1, Decimal("1.8"), Decimal("1.2")),
    "ROCHE": (SponsorReliability.TIER_1, Decimal("1.6"), Decimal("1.0")),
    "NOVARTIS": (SponsorReliability.TIER_1, Decimal("2.0"), Decimal("1.3")),
    "LILLY": (SponsorReliability.TIER_1, Decimal("1.8"), Decimal("1.1")),
    "ABBVIE": (SponsorReliability.TIER_1, Decimal("1.5"), Decimal("0.9")),
    "AMGEN": (SponsorReliability.TIER_1, Decimal("1.7"), Decimal("1.0")),
    "GILEAD": (SponsorReliability.TIER_1, Decimal("1.4"), Decimal("0.8")),
    "REGENERON": (SponsorReliability.TIER_1, Decimal("1.6"), Decimal("0.9")),
    "VERTEX": (SponsorReliability.TIER_1, Decimal("1.3"), Decimal("0.7")),

    # Tier 2 - Mid-size biotech with reasonable execution
    "BIOGEN": (SponsorReliability.TIER_2, Decimal("3.0"), Decimal("2.0")),
    "ALNYLAM": (SponsorReliability.TIER_2, Decimal("2.5"), Decimal("1.5")),
    "SAREPTA": (SponsorReliability.TIER_2, Decimal("3.5"), Decimal("2.2")),
    "BIOMARIN": (SponsorReliability.TIER_2, Decimal("2.8"), Decimal("1.8")),
    "ALEXION": (SponsorReliability.TIER_2, Decimal("2.2"), Decimal("1.4")),
    "ARGENX": (SponsorReliability.TIER_2, Decimal("2.4"), Decimal("1.5")),

    # Tier 3 - Smaller biotech with execution challenges
    # (populated dynamically from track record)
}

# Default delay for unknown sponsors
DEFAULT_SPONSOR_DELAY = (SponsorReliability.UNKNOWN, Decimal("4.0"), Decimal("3.0"))


@dataclass
class SponsorDelayProfile:
    """Sponsor-specific delay characteristics."""
    sponsor_name: str
    tier: SponsorReliability = SponsorReliability.UNKNOWN
    avg_delay_months: Decimal = Decimal("4.0")
    std_dev_months: Decimal = Decimal("3.0")
    n_trials_tracked: int = 0
    last_updated: str = ""

    # Historical slippage events
    recent_slips: List[int] = field(default_factory=list)  # Days slipped per trial

    @property
    def delay_factor(self) -> Decimal:
        """
        Delay factor for timeline adjustment.

        1.0 = on-time
        1.2 = 20% longer than expected
        """
        if self.tier == SponsorReliability.TIER_1:
            return Decimal("1.05")  # 5% buffer
        elif self.tier == SponsorReliability.TIER_2:
            return Decimal("1.15")  # 15% buffer
        else:
            return Decimal("1.25")  # 25% buffer

    @classmethod
    def from_database(cls, sponsor_name: str) -> "SponsorDelayProfile":
        """Load sponsor profile from database."""
        normalized_name = sponsor_name.upper().strip()

        if normalized_name in SPONSOR_DELAY_DATABASE:
            tier, avg, std = SPONSOR_DELAY_DATABASE[normalized_name]
        else:
            tier, avg, std = DEFAULT_SPONSOR_DELAY

        return cls(
            sponsor_name=sponsor_name,
            tier=tier,
            avg_delay_months=avg,
            std_dev_months=std,
        )


# =============================================================================
# PDUFA DATE TRACKING
# =============================================================================

class PDUFAType(str, Enum):
    """PDUFA action date types."""
    STANDARD = "STANDARD"
    PRIORITY = "PRIORITY"
    ACCELERATED = "ACCELERATED"
    BREAKTHROUGH = "BREAKTHROUGH"
    REAL_TIME = "REAL_TIME"


@dataclass
class PDUFADate:
    """PDUFA action date with metadata."""
    ticker: str
    drug_name: str
    indication: str
    pdufa_type: PDUFAType
    action_date: str  # ISO date
    submission_date: Optional[str] = None
    is_confirmed: bool = False
    source: str = "COMPANY"  # COMPANY, SEC, FDA

    @property
    def pdufa_id(self) -> str:
        """Stable PDUFA ID."""
        canonical = f"{self.ticker}|{self.drug_name}|{self.action_date}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def days_until(self, as_of_date: date) -> Optional[int]:
        """Days until PDUFA date from as_of_date."""
        try:
            action_d = date.fromisoformat(self.action_date)
            return (action_d - as_of_date).days
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdufa_id": self.pdufa_id,
            "ticker": self.ticker,
            "drug_name": self.drug_name,
            "indication": self.indication,
            "pdufa_type": self.pdufa_type.value,
            "action_date": self.action_date,
            "submission_date": self.submission_date,
            "is_confirmed": self.is_confirmed,
            "source": self.source,
        }


# =============================================================================
# TIMING ESTIMATE
# =============================================================================

class TimingConfidence(str, Enum):
    """Confidence level for timing estimate."""
    HIGH = "HIGH"  # <3 months uncertainty
    MEDIUM = "MEDIUM"  # 3-6 months uncertainty
    LOW = "LOW"  # 6-12 months uncertainty
    SPECULATIVE = "SPECULATIVE"  # >12 months uncertainty


@dataclass
class TimingEstimate:
    """Expected readout timing estimate."""
    ticker: str
    nct_id: str
    as_of_date: str

    # Enrollment data
    target_enrollment: int = 0
    current_enrollment: int = 0
    enrollment_rate_per_month: Decimal = Decimal("0")
    last_enrollment_update: Optional[str] = None

    # Timeline data
    phase: str = "UNKNOWN"
    expected_primary_completion: Optional[str] = None
    expected_study_completion: Optional[str] = None
    expected_results_date: Optional[str] = None

    # Sponsor adjustment
    sponsor_name: str = ""
    sponsor_delay_factor: Decimal = Decimal("1.0")

    # Computed estimates
    estimated_readout_date: Optional[str] = None
    estimated_readout_days: Optional[int] = None
    confidence: TimingConfidence = TimingConfidence.SPECULATIVE
    confidence_interval_days: Tuple[int, int] = (0, 0)

    # Audit
    calculation_log: List[str] = field(default_factory=list)

    @property
    def estimate_id(self) -> str:
        """Stable estimate ID."""
        canonical = f"{self.ticker}|{self.nct_id}|{self.as_of_date}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "ticker": self.ticker,
            "nct_id": self.nct_id,
            "as_of_date": self.as_of_date,
            "target_enrollment": self.target_enrollment,
            "current_enrollment": self.current_enrollment,
            "enrollment_rate_per_month": str(self.enrollment_rate_per_month),
            "phase": self.phase,
            "expected_primary_completion": self.expected_primary_completion,
            "sponsor_name": self.sponsor_name,
            "sponsor_delay_factor": str(self.sponsor_delay_factor),
            "estimated_readout_date": self.estimated_readout_date,
            "estimated_readout_days": self.estimated_readout_days,
            "confidence": self.confidence.value,
            "confidence_interval_days": list(self.confidence_interval_days),
            "calculation_log": self.calculation_log,
        }


# =============================================================================
# CATALYST CLUSTERING
# =============================================================================

@dataclass
class CatalystCluster:
    """Group of catalysts within a convexity window."""
    ticker: str
    window_start: str  # ISO date
    window_end: str  # ISO date
    n_catalysts: int
    catalyst_types: List[str]
    total_impact_score: Decimal = Decimal("0")
    is_convex: bool = False  # Multiple catalysts = higher total value

    @property
    def window_days(self) -> int:
        """Window size in days."""
        try:
            start = date.fromisoformat(self.window_start)
            end = date.fromisoformat(self.window_end)
            return (end - start).days
        except (ValueError, TypeError):
            return 0

    @property
    def cluster_id(self) -> str:
        """Stable cluster ID."""
        canonical = f"{self.ticker}|{self.window_start}|{self.window_end}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "ticker": self.ticker,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_days": self.window_days,
            "n_catalysts": self.n_catalysts,
            "catalyst_types": self.catalyst_types,
            "total_impact_score": str(self.total_impact_score),
            "is_convex": self.is_convex,
        }


# =============================================================================
# TIMING ENGINE
# =============================================================================

class TimingEngine:
    """
    Timing engine for catalyst readout estimation.

    Features:
    - Enrollment-based readout forecasting
    - Sponsor delay factor adjustment
    - PDUFA date tracking
    - Catalyst clustering detection
    """

    # Clustering parameters
    CLUSTER_WINDOW_DAYS = 90  # Catalysts within 90 days are clustered
    CONVEXITY_THRESHOLD = 2  # 2+ catalysts = convex cluster

    def __init__(self):
        self.pdufa_dates: Dict[str, List[PDUFADate]] = {}  # ticker -> PDUFAs
        self.sponsor_profiles: Dict[str, SponsorDelayProfile] = {}

    def load_sponsor_profile(self, sponsor_name: str) -> SponsorDelayProfile:
        """Load or create sponsor delay profile."""
        if sponsor_name not in self.sponsor_profiles:
            self.sponsor_profiles[sponsor_name] = SponsorDelayProfile.from_database(
                sponsor_name
            )
        return self.sponsor_profiles[sponsor_name]

    def estimate_enrollment_completion(
        self,
        current_enrollment: int,
        target_enrollment: int,
        rate_per_month: Decimal,
        as_of_date: date,
    ) -> Tuple[Optional[date], List[str]]:
        """
        Estimate enrollment completion date.

        Returns:
            (estimated_date, calculation_log)
        """
        log = []

        if current_enrollment >= target_enrollment:
            log.append(f"Enrollment complete: {current_enrollment}/{target_enrollment}")
            return (as_of_date, log)

        if rate_per_month <= 0:
            log.append("Cannot estimate: no enrollment rate")
            return (None, log)

        remaining = target_enrollment - current_enrollment
        months_remaining = Decimal(remaining) / rate_per_month

        log.append(f"Remaining enrollment: {remaining}")
        log.append(f"Rate: {rate_per_month}/month")
        log.append(f"Estimated months: {months_remaining:.1f}")

        days_remaining = int(months_remaining * Decimal("30.44"))
        estimated_date = as_of_date + timedelta(days=days_remaining)

        return (estimated_date, log)

    def estimate_readout_date(
        self,
        ticker: str,
        nct_id: str,
        phase: str,
        as_of_date: date,
        target_enrollment: int = 0,
        current_enrollment: int = 0,
        enrollment_rate_per_month: Decimal = Decimal("0"),
        expected_primary_completion: Optional[str] = None,
        sponsor_name: str = "",
    ) -> TimingEstimate:
        """
        Estimate catalyst readout date.

        Uses enrollment-based forecasting when available,
        falls back to PCD with sponsor adjustment.
        """
        calculation_log = []
        estimate = TimingEstimate(
            ticker=ticker,
            nct_id=nct_id,
            as_of_date=as_of_date.isoformat(),
            target_enrollment=target_enrollment,
            current_enrollment=current_enrollment,
            enrollment_rate_per_month=enrollment_rate_per_month,
            phase=phase,
            expected_primary_completion=expected_primary_completion,
            sponsor_name=sponsor_name,
        )

        # Get sponsor delay factor
        if sponsor_name:
            profile = self.load_sponsor_profile(sponsor_name)
            estimate.sponsor_delay_factor = profile.delay_factor
            calculation_log.append(
                f"Sponsor {sponsor_name} ({profile.tier.value}): delay factor {profile.delay_factor}"
            )
        else:
            calculation_log.append("No sponsor specified, using default delay factor")

        # Method 1: Enrollment-based estimation
        enrollment_date = None
        if target_enrollment > 0 and current_enrollment > 0 and enrollment_rate_per_month > 0:
            enrollment_date, enroll_log = self.estimate_enrollment_completion(
                current_enrollment,
                target_enrollment,
                enrollment_rate_per_month,
                as_of_date,
            )
            calculation_log.extend(enroll_log)

            if enrollment_date:
                # Add phase-specific lead time
                lead_time_months = READOUT_LEAD_TIMES.get(phase, 18)
                lead_time_days = int(lead_time_months * 30.44)

                # Apply sponsor delay factor
                adjusted_days = int(lead_time_days * float(estimate.sponsor_delay_factor))

                readout_date = enrollment_date + timedelta(days=adjusted_days)
                calculation_log.append(
                    f"Lead time: {lead_time_months} months × {estimate.sponsor_delay_factor} = {adjusted_days} days"
                )

                estimate.estimated_readout_date = readout_date.isoformat()
                estimate.estimated_readout_days = (readout_date - as_of_date).days

        # Method 2: PCD-based estimation (fallback)
        if estimate.estimated_readout_date is None and expected_primary_completion:
            try:
                pcd = date.fromisoformat(expected_primary_completion)
                # Apply sponsor delay factor
                days_to_pcd = (pcd - as_of_date).days
                adjusted_days = int(days_to_pcd * float(estimate.sponsor_delay_factor))
                readout_date = as_of_date + timedelta(days=adjusted_days)

                calculation_log.append(
                    f"Using PCD: {expected_primary_completion} × {estimate.sponsor_delay_factor}"
                )

                estimate.estimated_readout_date = readout_date.isoformat()
                estimate.estimated_readout_days = adjusted_days
            except (ValueError, TypeError):
                calculation_log.append(f"Invalid PCD: {expected_primary_completion}")

        # Compute confidence
        if estimate.estimated_readout_days is not None:
            days = estimate.estimated_readout_days
            if days <= 90:
                estimate.confidence = TimingConfidence.HIGH
                ci_range = 30
            elif days <= 180:
                estimate.confidence = TimingConfidence.MEDIUM
                ci_range = 60
            elif days <= 365:
                estimate.confidence = TimingConfidence.LOW
                ci_range = 120
            else:
                estimate.confidence = TimingConfidence.SPECULATIVE
                ci_range = 180

            estimate.confidence_interval_days = (
                max(0, days - ci_range),
                days + ci_range,
            )

        estimate.calculation_log = calculation_log
        return estimate

    def detect_clusters(
        self,
        catalyst_dates: List[Tuple[str, str, Decimal]],  # (date, type, impact)
        ticker: str,
    ) -> List[CatalystCluster]:
        """
        Detect catalyst clusters within convexity windows.

        Args:
            catalyst_dates: List of (date_iso, catalyst_type, impact_score)
            ticker: Ticker symbol

        Returns:
            List of CatalystCluster objects
        """
        if len(catalyst_dates) < 2:
            return []

        # Sort by date
        sorted_catalysts = sorted(catalyst_dates, key=lambda x: x[0])
        clusters = []
        current_cluster: List[Tuple[str, str, Decimal]] = []

        for i, (cat_date, cat_type, impact) in enumerate(sorted_catalysts):
            if not current_cluster:
                current_cluster.append((cat_date, cat_type, impact))
                continue

            # Check if within window of cluster start
            try:
                cluster_start = date.fromisoformat(current_cluster[0][0])
                current_date = date.fromisoformat(cat_date)
                days_from_start = (current_date - cluster_start).days

                if days_from_start <= self.CLUSTER_WINDOW_DAYS:
                    current_cluster.append((cat_date, cat_type, impact))
                else:
                    # Save current cluster if it meets threshold
                    if len(current_cluster) >= self.CONVEXITY_THRESHOLD:
                        cluster = self._create_cluster(current_cluster, ticker)
                        clusters.append(cluster)

                    # Start new cluster
                    current_cluster = [(cat_date, cat_type, impact)]
            except (ValueError, TypeError):
                continue

        # Check final cluster
        if len(current_cluster) >= self.CONVEXITY_THRESHOLD:
            cluster = self._create_cluster(current_cluster, ticker)
            clusters.append(cluster)

        return clusters

    def _create_cluster(
        self,
        catalysts: List[Tuple[str, str, Decimal]],
        ticker: str,
    ) -> CatalystCluster:
        """Create cluster from list of catalysts."""
        dates = [c[0] for c in catalysts]
        types = [c[1] for c in catalysts]
        impacts = [c[2] for c in catalysts]

        total_impact = sum(impacts, Decimal("0"))

        # Convexity: cluster value > sum of parts
        # Multiple catalysts in short window = higher optionality value
        convexity_bonus = Decimal("1.0") + (Decimal(len(catalysts) - 1) * Decimal("0.15"))
        convex_impact = total_impact * convexity_bonus

        return CatalystCluster(
            ticker=ticker,
            window_start=min(dates),
            window_end=max(dates),
            n_catalysts=len(catalysts),
            catalyst_types=sorted(set(types)),
            total_impact_score=convex_impact.quantize(Decimal("0.01")),
            is_convex=len(catalysts) >= self.CONVEXITY_THRESHOLD,
        )

    def add_pdufa_date(self, pdufa: PDUFADate) -> None:
        """Add PDUFA date to tracking."""
        if pdufa.ticker not in self.pdufa_dates:
            self.pdufa_dates[pdufa.ticker] = []
        self.pdufa_dates[pdufa.ticker].append(pdufa)

    def get_upcoming_pdufas(
        self,
        ticker: str,
        as_of_date: date,
        horizon_days: int = 365,
    ) -> List[PDUFADate]:
        """Get upcoming PDUFA dates for ticker."""
        pdufas = self.pdufa_dates.get(ticker, [])
        upcoming = []

        for pdufa in pdufas:
            days_until = pdufa.days_until(as_of_date)
            if days_until is not None and 0 <= days_until <= horizon_days:
                upcoming.append(pdufa)

        return sorted(upcoming, key=lambda p: p.action_date)

    def batch_estimate_timing(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, TimingEstimate]:
        """
        Batch estimate timing for multiple trials.

        Args:
            trials: List of trial dicts
            as_of_date: Point-in-time date

        Returns:
            {nct_id: TimingEstimate}
        """
        estimates = {}

        for trial in sorted(trials, key=lambda t: (t.get("ticker", ""), t.get("nct_id", ""))):
            ticker = trial.get("ticker", "")
            nct_id = trial.get("nct_id", "")
            phase = trial.get("phase", "UNKNOWN")

            estimate = self.estimate_readout_date(
                ticker=ticker,
                nct_id=nct_id,
                phase=phase,
                as_of_date=as_of_date,
                target_enrollment=trial.get("target_enrollment", 0),
                current_enrollment=trial.get("current_enrollment", 0),
                enrollment_rate_per_month=Decimal(str(trial.get("enrollment_rate", 0))),
                expected_primary_completion=trial.get("primary_completion_date"),
                sponsor_name=trial.get("sponsor", ""),
            )

            estimates[nct_id] = estimate

        return estimates


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_enrollment_rate(
    enrollment_history: List[Tuple[str, int]],  # (date, cumulative)
) -> Optional[Decimal]:
    """
    Estimate enrollment rate from historical data.

    Uses linear regression on recent data points.
    """
    if len(enrollment_history) < 2:
        return None

    # Sort by date
    sorted_history = sorted(enrollment_history, key=lambda x: x[0])

    # Use last 6 months of data
    recent = sorted_history[-6:] if len(sorted_history) > 6 else sorted_history

    try:
        first_date = date.fromisoformat(recent[0][0])
        last_date = date.fromisoformat(recent[-1][0])
        first_count = recent[0][1]
        last_count = recent[-1][1]

        days = (last_date - first_date).days
        if days <= 0:
            return None

        enrolled = last_count - first_count
        months = Decimal(days) / Decimal("30.44")

        if months <= 0:
            return None

        rate = Decimal(enrolled) / months
        return rate.quantize(Decimal("0.1"))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def canonical_json_dumps(obj: Any) -> str:
    """Serialize to canonical JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
