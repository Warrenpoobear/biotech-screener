#!/usr/bin/env python3
"""
catalyst_probability_engine.py - Probability of Success Engine for Catalyst Module

Implements Bayesian probability calibration using:
- FDA historical base rates by phase and indication
- Trial Design Quality Score (TDQS)
- Competitive landscape adjustments
- Sponsor track record
- Time decay for stale estimates

Design Philosophy:
- DETERMINISTIC: No randomness, reproducible outputs
- STDLIB-ONLY: No external dependencies
- PIT-SAFE: All dates explicit
- GOVERNED: Auditable probability updates

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json


# =============================================================================
# FDA BASE RATES BY PHASE (Historical Approval Rates)
# =============================================================================
# Source: BIO/Informa Pharma Intelligence Clinical Development Success Rates
# Updated with 2020-2024 data

class Phase(str, Enum):
    """Clinical trial phases."""
    PRECLINICAL = "PRECLINICAL"
    P1 = "P1"
    P1_2 = "P1_2"
    P2 = "P2"
    P2_3 = "P2_3"
    P3 = "P3"
    NDA_BLA = "NDA_BLA"
    APPROVED = "APPROVED"
    UNKNOWN = "UNKNOWN"


class TherapeuticArea(str, Enum):
    """Therapeutic area categories."""
    ONCOLOGY = "ONCOLOGY"
    CNS = "CNS"
    CARDIOVASCULAR = "CARDIOVASCULAR"
    METABOLIC = "METABOLIC"
    INFECTIOUS_DISEASE = "INFECTIOUS_DISEASE"
    IMMUNOLOGY = "IMMUNOLOGY"
    RARE_DISEASE = "RARE_DISEASE"
    GENE_THERAPY = "GENE_THERAPY"
    VACCINES = "VACCINES"
    OTHER = "OTHER"


# FDA base rates: P(success | phase, therapeutic_area)
# Format: {phase: {therapeutic_area: probability}}
FDA_BASE_RATES: Dict[Phase, Dict[TherapeuticArea, Decimal]] = {
    Phase.P1: {
        TherapeuticArea.ONCOLOGY: Decimal("0.52"),
        TherapeuticArea.CNS: Decimal("0.48"),
        TherapeuticArea.CARDIOVASCULAR: Decimal("0.55"),
        TherapeuticArea.METABOLIC: Decimal("0.58"),
        TherapeuticArea.INFECTIOUS_DISEASE: Decimal("0.62"),
        TherapeuticArea.IMMUNOLOGY: Decimal("0.54"),
        TherapeuticArea.RARE_DISEASE: Decimal("0.65"),
        TherapeuticArea.GENE_THERAPY: Decimal("0.45"),
        TherapeuticArea.VACCINES: Decimal("0.68"),
        TherapeuticArea.OTHER: Decimal("0.52"),
    },
    Phase.P2: {
        TherapeuticArea.ONCOLOGY: Decimal("0.29"),
        TherapeuticArea.CNS: Decimal("0.18"),
        TherapeuticArea.CARDIOVASCULAR: Decimal("0.25"),
        TherapeuticArea.METABOLIC: Decimal("0.35"),
        TherapeuticArea.INFECTIOUS_DISEASE: Decimal("0.42"),
        TherapeuticArea.IMMUNOLOGY: Decimal("0.32"),
        TherapeuticArea.RARE_DISEASE: Decimal("0.48"),
        TherapeuticArea.GENE_THERAPY: Decimal("0.35"),
        TherapeuticArea.VACCINES: Decimal("0.55"),
        TherapeuticArea.OTHER: Decimal("0.30"),
    },
    Phase.P3: {
        TherapeuticArea.ONCOLOGY: Decimal("0.40"),
        TherapeuticArea.CNS: Decimal("0.48"),
        TherapeuticArea.CARDIOVASCULAR: Decimal("0.52"),
        TherapeuticArea.METABOLIC: Decimal("0.58"),
        TherapeuticArea.INFECTIOUS_DISEASE: Decimal("0.62"),
        TherapeuticArea.IMMUNOLOGY: Decimal("0.55"),
        TherapeuticArea.RARE_DISEASE: Decimal("0.68"),
        TherapeuticArea.GENE_THERAPY: Decimal("0.55"),
        TherapeuticArea.VACCINES: Decimal("0.72"),
        TherapeuticArea.OTHER: Decimal("0.50"),
    },
    Phase.NDA_BLA: {
        TherapeuticArea.ONCOLOGY: Decimal("0.85"),
        TherapeuticArea.CNS: Decimal("0.88"),
        TherapeuticArea.CARDIOVASCULAR: Decimal("0.90"),
        TherapeuticArea.METABOLIC: Decimal("0.92"),
        TherapeuticArea.INFECTIOUS_DISEASE: Decimal("0.90"),
        TherapeuticArea.IMMUNOLOGY: Decimal("0.88"),
        TherapeuticArea.RARE_DISEASE: Decimal("0.92"),
        TherapeuticArea.GENE_THERAPY: Decimal("0.80"),
        TherapeuticArea.VACCINES: Decimal("0.95"),
        TherapeuticArea.OTHER: Decimal("0.85"),
    },
}

# Default rates for phases not in FDA_BASE_RATES
DEFAULT_BASE_RATE = Decimal("0.50")


# =============================================================================
# TRIAL DESIGN QUALITY SCORE (TDQS)
# =============================================================================

class TrialDesign(str, Enum):
    """Trial design types."""
    RANDOMIZED_CONTROLLED = "RANDOMIZED_CONTROLLED"
    DOUBLE_BLIND = "DOUBLE_BLIND"
    OPEN_LABEL = "OPEN_LABEL"
    SINGLE_ARM = "SINGLE_ARM"
    ADAPTIVE = "ADAPTIVE"
    BASKET = "BASKET"
    UMBRELLA = "UMBRELLA"
    PLATFORM = "PLATFORM"
    CROSSOVER = "CROSSOVER"
    UNKNOWN = "UNKNOWN"


class EndpointType(str, Enum):
    """Primary endpoint types."""
    OVERALL_SURVIVAL = "OS"
    PROGRESSION_FREE_SURVIVAL = "PFS"
    OBJECTIVE_RESPONSE_RATE = "ORR"
    COMPLETE_RESPONSE = "CR"
    DISEASE_FREE_SURVIVAL = "DFS"
    DURATION_OF_RESPONSE = "DOR"
    CLINICAL_BENEFIT_RATE = "CBR"
    BIOMARKER = "BIOMARKER"
    PATIENT_REPORTED = "PRO"
    SAFETY = "SAFETY"
    OTHER = "OTHER"


# TDQS component weights
TDQS_WEIGHTS = {
    "design": Decimal("0.30"),
    "endpoint": Decimal("0.25"),
    "sample_size": Decimal("0.20"),
    "control_arm": Decimal("0.15"),
    "biomarker_selection": Decimal("0.10"),
}

# Design quality scores
DESIGN_SCORES: Dict[TrialDesign, Decimal] = {
    TrialDesign.RANDOMIZED_CONTROLLED: Decimal("1.0"),
    TrialDesign.DOUBLE_BLIND: Decimal("1.0"),
    TrialDesign.ADAPTIVE: Decimal("0.95"),
    TrialDesign.PLATFORM: Decimal("0.90"),
    TrialDesign.CROSSOVER: Decimal("0.85"),
    TrialDesign.OPEN_LABEL: Decimal("0.70"),
    TrialDesign.BASKET: Decimal("0.65"),
    TrialDesign.UMBRELLA: Decimal("0.65"),
    TrialDesign.SINGLE_ARM: Decimal("0.50"),
    TrialDesign.UNKNOWN: Decimal("0.40"),
}

# Endpoint quality scores
ENDPOINT_SCORES: Dict[EndpointType, Decimal] = {
    EndpointType.OVERALL_SURVIVAL: Decimal("1.0"),
    EndpointType.PROGRESSION_FREE_SURVIVAL: Decimal("0.90"),
    EndpointType.DISEASE_FREE_SURVIVAL: Decimal("0.85"),
    EndpointType.COMPLETE_RESPONSE: Decimal("0.80"),
    EndpointType.OBJECTIVE_RESPONSE_RATE: Decimal("0.75"),
    EndpointType.DURATION_OF_RESPONSE: Decimal("0.70"),
    EndpointType.CLINICAL_BENEFIT_RATE: Decimal("0.65"),
    EndpointType.PATIENT_REPORTED: Decimal("0.60"),
    EndpointType.BIOMARKER: Decimal("0.55"),
    EndpointType.SAFETY: Decimal("0.50"),
    EndpointType.OTHER: Decimal("0.40"),
}


@dataclass
class TrialDesignProfile:
    """Trial design characteristics for TDQS calculation."""
    design_type: TrialDesign = TrialDesign.UNKNOWN
    primary_endpoint: EndpointType = EndpointType.OTHER
    target_enrollment: int = 0
    actual_enrollment: int = 0
    has_control_arm: bool = False
    has_biomarker_selection: bool = False
    is_registrational: bool = False
    has_breakthrough_designation: bool = False
    has_fast_track: bool = False
    has_priority_review: bool = False

    def compute_tdqs(self) -> Decimal:
        """Compute Trial Design Quality Score (0-100)."""
        # Design score
        design_score = DESIGN_SCORES.get(self.design_type, Decimal("0.40"))

        # Endpoint score
        endpoint_score = ENDPOINT_SCORES.get(self.primary_endpoint, Decimal("0.40"))

        # Sample size score (enrollment adequacy)
        if self.target_enrollment <= 0:
            sample_score = Decimal("0.50")
        elif self.actual_enrollment >= self.target_enrollment:
            sample_score = Decimal("1.0")
        elif self.actual_enrollment >= self.target_enrollment * 0.8:
            sample_score = Decimal("0.85")
        elif self.actual_enrollment >= self.target_enrollment * 0.5:
            sample_score = Decimal("0.65")
        else:
            sample_score = Decimal("0.40")

        # Control arm score
        control_score = Decimal("1.0") if self.has_control_arm else Decimal("0.50")

        # Biomarker selection score
        biomarker_score = Decimal("1.0") if self.has_biomarker_selection else Decimal("0.70")

        # Weighted sum
        tdqs = (
            TDQS_WEIGHTS["design"] * design_score +
            TDQS_WEIGHTS["endpoint"] * endpoint_score +
            TDQS_WEIGHTS["sample_size"] * sample_score +
            TDQS_WEIGHTS["control_arm"] * control_score +
            TDQS_WEIGHTS["biomarker_selection"] * biomarker_score
        )

        # Bonus for regulatory designations
        designation_bonus = Decimal("0")
        if self.has_breakthrough_designation:
            designation_bonus += Decimal("0.10")
        if self.has_fast_track:
            designation_bonus += Decimal("0.05")
        if self.has_priority_review:
            designation_bonus += Decimal("0.05")
        if self.is_registrational:
            designation_bonus += Decimal("0.05")

        final_tdqs = min(Decimal("1.0"), tdqs + designation_bonus)
        return (final_tdqs * 100).quantize(Decimal("0.1"))


# =============================================================================
# COMPETITIVE LANDSCAPE
# =============================================================================

@dataclass
class CompetitiveLandscape:
    """Competitive landscape for an indication."""
    indication: str
    n_approved_therapies: int = 0
    n_phase3_competitors: int = 0
    n_phase2_competitors: int = 0
    has_standard_of_care: bool = True
    is_first_in_class: bool = False
    is_best_in_class_potential: bool = False
    unmet_need_level: str = "MODERATE"  # HIGH, MODERATE, LOW

    def compute_competitive_adjustment(self) -> Decimal:
        """
        Compute probability adjustment based on competitive landscape.

        Returns multiplier (0.5 - 1.5) applied to base probability.
        """
        adjustment = Decimal("1.0")

        # Crowded market penalty
        if self.n_approved_therapies > 5:
            adjustment -= Decimal("0.15")
        elif self.n_approved_therapies > 2:
            adjustment -= Decimal("0.08")

        # Phase 3 competition penalty
        if self.n_phase3_competitors > 3:
            adjustment -= Decimal("0.10")
        elif self.n_phase3_competitors > 1:
            adjustment -= Decimal("0.05")

        # First-in-class bonus
        if self.is_first_in_class:
            adjustment += Decimal("0.15")

        # Best-in-class potential bonus
        if self.is_best_in_class_potential:
            adjustment += Decimal("0.10")

        # Unmet need bonus
        if self.unmet_need_level == "HIGH":
            adjustment += Decimal("0.20")
        elif self.unmet_need_level == "MODERATE":
            adjustment += Decimal("0.05")

        # Clamp to reasonable range
        return max(Decimal("0.50"), min(Decimal("1.50"), adjustment))


# =============================================================================
# SPONSOR TRACK RECORD
# =============================================================================

@dataclass
class SponsorTrackRecord:
    """Historical track record for a sponsor."""
    sponsor_name: str
    n_approvals_5yr: int = 0
    n_failures_5yr: int = 0
    n_crl_5yr: int = 0
    avg_enrollment_delay_months: Decimal = Decimal("0")
    avg_timeline_slip_months: Decimal = Decimal("0")
    has_manufacturing_issues: bool = False
    has_regulatory_warning: bool = False

    @property
    def success_rate(self) -> Decimal:
        """Historical success rate."""
        total = self.n_approvals_5yr + self.n_failures_5yr
        if total == 0:
            return Decimal("0.50")  # No track record = neutral
        return Decimal(self.n_approvals_5yr) / Decimal(total)

    def compute_sponsor_adjustment(self) -> Decimal:
        """
        Compute probability adjustment based on sponsor track record.

        Returns multiplier (0.7 - 1.3) applied to base probability.
        """
        adjustment = Decimal("1.0")

        # Success rate adjustment
        sr = self.success_rate
        if sr > Decimal("0.70"):
            adjustment += Decimal("0.15")
        elif sr > Decimal("0.50"):
            adjustment += Decimal("0.05")
        elif sr < Decimal("0.30"):
            adjustment -= Decimal("0.15")
        elif sr < Decimal("0.50"):
            adjustment -= Decimal("0.05")

        # CRL penalty
        if self.n_crl_5yr > 2:
            adjustment -= Decimal("0.10")
        elif self.n_crl_5yr > 0:
            adjustment -= Decimal("0.05")

        # Execution issues
        if self.has_manufacturing_issues:
            adjustment -= Decimal("0.10")
        if self.has_regulatory_warning:
            adjustment -= Decimal("0.05")

        # Timeline reliability
        if self.avg_timeline_slip_months > Decimal("12"):
            adjustment -= Decimal("0.08")
        elif self.avg_timeline_slip_months > Decimal("6"):
            adjustment -= Decimal("0.03")

        return max(Decimal("0.70"), min(Decimal("1.30"), adjustment))


# =============================================================================
# PROBABILITY ENGINE
# =============================================================================

@dataclass
class ProbabilityEstimate:
    """Probability of success estimate with full audit trail."""
    ticker: str
    nct_id: str
    as_of_date: str

    # Base probability
    phase: Phase = Phase.UNKNOWN
    therapeutic_area: TherapeuticArea = TherapeuticArea.OTHER
    base_probability: Decimal = Decimal("0.50")

    # Adjustments
    tdqs: Decimal = Decimal("50.0")
    tdqs_multiplier: Decimal = Decimal("1.0")
    competitive_multiplier: Decimal = Decimal("1.0")
    sponsor_multiplier: Decimal = Decimal("1.0")
    time_decay_multiplier: Decimal = Decimal("1.0")

    # Final probability
    adjusted_probability: Decimal = Decimal("0.50")
    confidence_interval_low: Decimal = Decimal("0.30")
    confidence_interval_high: Decimal = Decimal("0.70")

    # Audit
    adjustment_log: List[str] = field(default_factory=list)

    @property
    def estimate_id(self) -> str:
        """Stable estimate ID."""
        canonical = f"{self.ticker}|{self.nct_id}|{self.as_of_date}|{self.adjusted_probability}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "ticker": self.ticker,
            "nct_id": self.nct_id,
            "as_of_date": self.as_of_date,
            "phase": self.phase.value,
            "therapeutic_area": self.therapeutic_area.value,
            "base_probability": str(self.base_probability),
            "tdqs": str(self.tdqs),
            "tdqs_multiplier": str(self.tdqs_multiplier),
            "competitive_multiplier": str(self.competitive_multiplier),
            "sponsor_multiplier": str(self.sponsor_multiplier),
            "time_decay_multiplier": str(self.time_decay_multiplier),
            "adjusted_probability": str(self.adjusted_probability),
            "confidence_interval": [
                str(self.confidence_interval_low),
                str(self.confidence_interval_high),
            ],
            "adjustment_log": self.adjustment_log,
        }


class ProbabilityEngine:
    """
    Bayesian probability engine for catalyst success estimation.

    Combines:
    - FDA historical base rates
    - Trial Design Quality Score (TDQS)
    - Competitive landscape adjustments
    - Sponsor track record
    - Time decay for stale estimates
    """

    # Time decay parameters
    ESTIMATE_HALF_LIFE_DAYS = 180
    STALENESS_THRESHOLD_DAYS = 365

    def __init__(self):
        self.sponsor_cache: Dict[str, SponsorTrackRecord] = {}
        self.competitive_cache: Dict[str, CompetitiveLandscape] = {}

    def get_base_rate(
        self,
        phase: Phase,
        therapeutic_area: TherapeuticArea,
    ) -> Decimal:
        """Get FDA base rate for phase and therapeutic area."""
        phase_rates = FDA_BASE_RATES.get(phase)
        if phase_rates is None:
            return DEFAULT_BASE_RATE
        return phase_rates.get(therapeutic_area, DEFAULT_BASE_RATE)

    def compute_tdqs_multiplier(self, tdqs: Decimal) -> Decimal:
        """
        Convert TDQS score to probability multiplier.

        TDQS 100 → 1.2x multiplier
        TDQS 50 → 1.0x multiplier (neutral)
        TDQS 0 → 0.8x multiplier
        """
        # Linear interpolation
        normalized = (tdqs - Decimal("50")) / Decimal("50")
        multiplier = Decimal("1.0") + (normalized * Decimal("0.20"))
        return max(Decimal("0.80"), min(Decimal("1.20"), multiplier))

    def compute_time_decay(
        self,
        estimate_date: date,
        as_of_date: date,
    ) -> Decimal:
        """
        Compute time decay multiplier for stale estimates.

        Recent estimates get full weight, older estimates decay.
        """
        days_old = (as_of_date - estimate_date).days

        if days_old <= 0:
            return Decimal("1.0")

        if days_old > self.STALENESS_THRESHOLD_DAYS:
            return Decimal("0.70")  # Minimum weight for very stale

        # Exponential decay
        decay_factor = Decimal(days_old) / Decimal(self.ESTIMATE_HALF_LIFE_DAYS)
        multiplier = Decimal("0.5") ** decay_factor

        return max(Decimal("0.70"), multiplier.quantize(Decimal("0.001")))

    def estimate_probability(
        self,
        ticker: str,
        nct_id: str,
        phase: Phase,
        therapeutic_area: TherapeuticArea,
        as_of_date: date,
        trial_design: Optional[TrialDesignProfile] = None,
        competitive_landscape: Optional[CompetitiveLandscape] = None,
        sponsor_track_record: Optional[SponsorTrackRecord] = None,
        estimate_date: Optional[date] = None,
    ) -> ProbabilityEstimate:
        """
        Compute probability of success estimate.

        Combines all factors into a single calibrated probability.
        """
        adjustment_log = []

        # 1. Get base rate
        base_prob = self.get_base_rate(phase, therapeutic_area)
        adjustment_log.append(
            f"Base rate ({phase.value}, {therapeutic_area.value}): {base_prob}"
        )

        # 2. TDQS adjustment
        if trial_design:
            tdqs = trial_design.compute_tdqs()
            tdqs_mult = self.compute_tdqs_multiplier(tdqs)
        else:
            tdqs = Decimal("50.0")
            tdqs_mult = Decimal("1.0")
        adjustment_log.append(f"TDQS: {tdqs} → multiplier {tdqs_mult}")

        # 3. Competitive adjustment
        if competitive_landscape:
            comp_mult = competitive_landscape.compute_competitive_adjustment()
        else:
            comp_mult = Decimal("1.0")
        adjustment_log.append(f"Competitive multiplier: {comp_mult}")

        # 4. Sponsor adjustment
        if sponsor_track_record:
            sponsor_mult = sponsor_track_record.compute_sponsor_adjustment()
        else:
            sponsor_mult = Decimal("1.0")
        adjustment_log.append(f"Sponsor multiplier: {sponsor_mult}")

        # 5. Time decay
        if estimate_date:
            time_mult = self.compute_time_decay(estimate_date, as_of_date)
        else:
            time_mult = Decimal("1.0")
        adjustment_log.append(f"Time decay multiplier: {time_mult}")

        # 6. Combined probability
        adjusted_prob = base_prob * tdqs_mult * comp_mult * sponsor_mult * time_mult
        adjusted_prob = max(Decimal("0.01"), min(Decimal("0.99"), adjusted_prob))
        adjusted_prob = adjusted_prob.quantize(Decimal("0.001"))
        adjustment_log.append(f"Adjusted probability: {adjusted_prob}")

        # 7. Confidence interval (simple heuristic)
        uncertainty = Decimal("0.15")
        if trial_design and trial_design.has_breakthrough_designation:
            uncertainty = Decimal("0.10")

        ci_low = max(Decimal("0.01"), adjusted_prob - uncertainty)
        ci_high = min(Decimal("0.99"), adjusted_prob + uncertainty)

        return ProbabilityEstimate(
            ticker=ticker,
            nct_id=nct_id,
            as_of_date=as_of_date.isoformat(),
            phase=phase,
            therapeutic_area=therapeutic_area,
            base_probability=base_prob,
            tdqs=tdqs,
            tdqs_multiplier=tdqs_mult,
            competitive_multiplier=comp_mult,
            sponsor_multiplier=sponsor_mult,
            time_decay_multiplier=time_mult,
            adjusted_probability=adjusted_prob,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            adjustment_log=adjustment_log,
        )

    def batch_estimate(
        self,
        trials: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, ProbabilityEstimate]:
        """
        Batch compute probability estimates for multiple trials.

        Args:
            trials: List of trial dicts with keys:
                - ticker, nct_id, phase, therapeutic_area
                - Optional: trial_design, competitive_landscape, sponsor
            as_of_date: Point-in-time date

        Returns:
            {nct_id: ProbabilityEstimate}
        """
        estimates = {}

        for trial in sorted(trials, key=lambda t: (t.get("ticker", ""), t.get("nct_id", ""))):
            ticker = trial.get("ticker", "UNKNOWN")
            nct_id = trial.get("nct_id", "")

            # Parse phase
            phase_str = trial.get("phase", "UNKNOWN")
            try:
                phase = Phase(phase_str)
            except ValueError:
                phase = Phase.UNKNOWN

            # Parse therapeutic area
            ta_str = trial.get("therapeutic_area", "OTHER")
            try:
                therapeutic_area = TherapeuticArea(ta_str)
            except ValueError:
                therapeutic_area = TherapeuticArea.OTHER

            # Get trial design if available
            trial_design = trial.get("trial_design")
            if isinstance(trial_design, dict):
                trial_design = TrialDesignProfile(**trial_design)

            # Get competitive landscape if available
            competitive = trial.get("competitive_landscape")
            if isinstance(competitive, dict):
                competitive = CompetitiveLandscape(**competitive)

            # Get sponsor track record if available
            sponsor = trial.get("sponsor_track_record")
            if isinstance(sponsor, dict):
                sponsor = SponsorTrackRecord(**sponsor)

            estimate = self.estimate_probability(
                ticker=ticker,
                nct_id=nct_id,
                phase=phase,
                therapeutic_area=therapeutic_area,
                as_of_date=as_of_date,
                trial_design=trial_design,
                competitive_landscape=competitive,
                sponsor_track_record=sponsor,
            )

            estimates[nct_id] = estimate

        return estimates


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_phase(phase_str: str) -> Phase:
    """Parse phase string to Phase enum."""
    phase_map = {
        "1": Phase.P1,
        "I": Phase.P1,
        "Phase 1": Phase.P1,
        "Phase I": Phase.P1,
        "1/2": Phase.P1_2,
        "I/II": Phase.P1_2,
        "Phase 1/2": Phase.P1_2,
        "Phase 1/Phase 2": Phase.P1_2,
        "2": Phase.P2,
        "II": Phase.P2,
        "Phase 2": Phase.P2,
        "Phase II": Phase.P2,
        "2/3": Phase.P2_3,
        "II/III": Phase.P2_3,
        "Phase 2/3": Phase.P2_3,
        "Phase 2/Phase 3": Phase.P2_3,
        "3": Phase.P3,
        "III": Phase.P3,
        "Phase 3": Phase.P3,
        "Phase III": Phase.P3,
        "NDA": Phase.NDA_BLA,
        "BLA": Phase.NDA_BLA,
        "NDA/BLA": Phase.NDA_BLA,
    }
    return phase_map.get(phase_str, Phase.UNKNOWN)


def parse_therapeutic_area(indication: str) -> TherapeuticArea:
    """Parse indication string to TherapeuticArea."""
    indication_lower = indication.lower()

    oncology_keywords = ["cancer", "tumor", "oncology", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma"]
    cns_keywords = ["alzheimer", "parkinson", "depression", "schizophrenia", "epilepsy", "migraine", "ms ", "multiple sclerosis"]
    cv_keywords = ["heart", "cardiac", "cardiovascular", "hypertension", "stroke", "atherosclerosis"]
    metabolic_keywords = ["diabetes", "obesity", "metabolic", "nash", "nafld", "lipid"]
    infectious_keywords = ["hiv", "hepatitis", "covid", "infection", "bacterial", "viral", "fungal"]
    immunology_keywords = ["rheumatoid", "lupus", "crohn", "colitis", "psoriasis", "atopic"]
    rare_keywords = ["rare", "orphan", "duchenne", "sma", "hemophilia"]
    gene_keywords = ["gene therapy", "crispr", "cell therapy", "car-t"]
    vaccine_keywords = ["vaccine", "immunization"]

    for kw in oncology_keywords:
        if kw in indication_lower:
            return TherapeuticArea.ONCOLOGY

    for kw in cns_keywords:
        if kw in indication_lower:
            return TherapeuticArea.CNS

    for kw in cv_keywords:
        if kw in indication_lower:
            return TherapeuticArea.CARDIOVASCULAR

    for kw in metabolic_keywords:
        if kw in indication_lower:
            return TherapeuticArea.METABOLIC

    for kw in infectious_keywords:
        if kw in indication_lower:
            return TherapeuticArea.INFECTIOUS_DISEASE

    for kw in immunology_keywords:
        if kw in indication_lower:
            return TherapeuticArea.IMMUNOLOGY

    for kw in rare_keywords:
        if kw in indication_lower:
            return TherapeuticArea.RARE_DISEASE

    for kw in gene_keywords:
        if kw in indication_lower:
            return TherapeuticArea.GENE_THERAPY

    for kw in vaccine_keywords:
        if kw in indication_lower:
            return TherapeuticArea.VACCINES

    return TherapeuticArea.OTHER


# =============================================================================
# CANONICAL JSON
# =============================================================================

def canonical_json_dumps(obj: Any) -> str:
    """Serialize to canonical JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
