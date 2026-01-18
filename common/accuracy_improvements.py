"""
Accuracy Improvement Utilities for Biotech Screener

This module contains 8 priority accuracy improvements:

1. Indication-specific endpoint weighting
2. Phase-dependent staleness thresholds
3. Regulatory pathway scoring
4. Regime-adaptive catalyst decay
5. Competitive landscape penalty
6. Dynamic dilution (VIX-adjusted)
7. Burn seasonality adjustment
8. Binary event proximity boost

Design Principles:
- Deterministic: No datetime.now(), no randomness
- Decimal arithmetic for financial precision
- PIT-safe: All functions require explicit as_of_date
- Fail-loud: Clear error states

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

__version__ = "1.0.0"
__all__ = [
    # Indication-specific endpoint weighting
    "TherapeuticArea",
    "EndpointWeightConfig",
    "classify_therapeutic_area",
    "compute_weighted_endpoint_score",
    # Phase-dependent staleness
    "PhaseStalenessConfig",
    "compute_phase_staleness",
    "get_staleness_threshold_for_phase",
    # Regulatory pathway scoring
    "RegulatoryDesignation",
    "RegulatoryPathwayScore",
    "compute_regulatory_pathway_score",
    # Regime-adaptive catalyst decay
    "MarketRegimeType",
    "RegimeDecayConfig",
    "compute_regime_adaptive_decay",
    # Competitive landscape penalty
    "CompetitiveLandscapeResult",
    "compute_competition_penalty",
    # Dynamic dilution
    "VixDilutionAdjustment",
    "compute_vix_dilution_adjustment",
    # Burn seasonality
    "QuarterlySeasonalityConfig",
    "compute_burn_seasonality_adjustment",
    # Binary event proximity boost
    "ProximityBoostResult",
    "compute_binary_event_proximity_boost",
]


# =============================================================================
# 1. INDICATION-SPECIFIC ENDPOINT WEIGHTING
# =============================================================================

class TherapeuticArea(str, Enum):
    """Therapeutic area classifications."""
    ONCOLOGY = "oncology"
    AUTOIMMUNE = "autoimmune"
    CNS = "cns"
    CARDIOVASCULAR = "cardiovascular"
    INFECTIOUS_DISEASE = "infectious_disease"
    RARE_DISEASE = "rare_disease"
    METABOLIC = "metabolic"
    RESPIRATORY = "respiratory"
    OPHTHALMOLOGY = "ophthalmology"
    DERMATOLOGY = "dermatology"
    OTHER = "other"


# Therapeutic area keywords for classification
THERAPEUTIC_AREA_KEYWORDS: Dict[TherapeuticArea, Set[str]] = {
    TherapeuticArea.ONCOLOGY: {
        "cancer", "tumor", "tumour", "carcinoma", "sarcoma", "leukemia", "lymphoma",
        "melanoma", "oncology", "malignant", "neoplasm", "metastatic", "glioblastoma",
        "myeloma", "adenocarcinoma", "nsclc", "sclc", "hcc", "rcc", "aml", "cll",
    },
    TherapeuticArea.AUTOIMMUNE: {
        "autoimmune", "rheumatoid", "lupus", "psoriasis", "crohn", "colitis",
        "multiple sclerosis", "arthritis", "inflammatory", "ibd", "sle",
        "ankylosing spondylitis", "dermatomyositis", "myasthenia",
    },
    TherapeuticArea.CNS: {
        "alzheimer", "parkinson", "depression", "anxiety", "schizophrenia",
        "epilepsy", "seizure", "cns", "neurological", "neuropathy", "migraine",
        "huntington", "als", "dementia", "bipolar", "adhd", "autism",
    },
    TherapeuticArea.CARDIOVASCULAR: {
        "cardiovascular", "heart", "cardiac", "hypertension", "atherosclerosis",
        "arrhythmia", "heart failure", "myocardial", "stroke", "thrombosis",
        "pulmonary hypertension", "angina", "atrial fibrillation",
    },
    TherapeuticArea.INFECTIOUS_DISEASE: {
        "infection", "infectious", "viral", "bacterial", "hiv", "hepatitis",
        "influenza", "covid", "antibiotic", "antiviral", "sepsis", "pneumonia",
        "tuberculosis", "fungal", "rsv", "ebola",
    },
    TherapeuticArea.RARE_DISEASE: {
        "orphan", "rare disease", "ultra-rare", "genetic disorder", "lysosomal",
        "duchenne", "sma", "spinal muscular", "hemophilia", "fabry", "gaucher",
        "pompe", "cystic fibrosis", "phenylketonuria", "pku",
    },
    TherapeuticArea.METABOLIC: {
        "diabetes", "obesity", "metabolic", "nash", "nafld", "hyperlipidemia",
        "hypercholesterolemia", "gout", "thyroid", "adrenal",
    },
    TherapeuticArea.RESPIRATORY: {
        "asthma", "copd", "respiratory", "pulmonary", "lung", "bronchitis",
        "idiopathic pulmonary fibrosis", "ipf", "cystic fibrosis",
    },
    TherapeuticArea.OPHTHALMOLOGY: {
        "eye", "ocular", "retinal", "macular", "glaucoma", "dry eye",
        "diabetic retinopathy", "amd", "uveitis", "ophthalmology",
    },
    TherapeuticArea.DERMATOLOGY: {
        "skin", "dermatology", "eczema", "atopic dermatitis", "acne",
        "vitiligo", "alopecia", "hidradenitis",
    },
}


# Endpoint weights by therapeutic area
# Values represent relative strength of endpoint (1.0 = gold standard)
ENDPOINT_WEIGHTS_BY_AREA: Dict[TherapeuticArea, Dict[str, Decimal]] = {
    TherapeuticArea.ONCOLOGY: {
        "overall_survival": Decimal("1.00"),
        "os": Decimal("1.00"),
        "progression_free_survival": Decimal("0.85"),
        "pfs": Decimal("0.85"),
        "disease_free_survival": Decimal("0.80"),
        "dfs": Decimal("0.80"),
        "event_free_survival": Decimal("0.80"),
        "efs": Decimal("0.80"),
        "complete_response": Decimal("0.70"),
        "cr": Decimal("0.70"),
        "objective_response_rate": Decimal("0.60"),
        "orr": Decimal("0.60"),
        "duration_of_response": Decimal("0.55"),
        "dor": Decimal("0.55"),
        "partial_response": Decimal("0.50"),
        "biomarker": Decimal("0.30"),
        "safety": Decimal("0.25"),
        "pk": Decimal("0.20"),
    },
    TherapeuticArea.AUTOIMMUNE: {
        "clinical_remission": Decimal("1.00"),
        "acr70": Decimal("0.95"),
        "acr50": Decimal("0.85"),
        "acr20": Decimal("0.70"),
        "das28": Decimal("0.75"),
        "pasi90": Decimal("0.90"),
        "pasi75": Decimal("0.80"),
        "pasi50": Decimal("0.65"),
        "crp_reduction": Decimal("0.50"),
        "esr_reduction": Decimal("0.45"),
        "biomarker": Decimal("0.40"),
        "safety": Decimal("0.25"),
    },
    TherapeuticArea.CNS: {
        "clinical_global_impression": Decimal("1.00"),
        "cgi": Decimal("1.00"),
        "functional_improvement": Decimal("0.90"),
        "cognitive_improvement": Decimal("0.85"),
        "mmse_change": Decimal("0.80"),
        "adas_cog": Decimal("0.80"),
        "seizure_reduction": Decimal("0.85"),
        "madrs_change": Decimal("0.75"),
        "hamd_change": Decimal("0.75"),
        "panss_change": Decimal("0.75"),
        "biomarker": Decimal("0.35"),
        "safety": Decimal("0.30"),
    },
    TherapeuticArea.CARDIOVASCULAR: {
        "mace": Decimal("1.00"),  # Major adverse cardiovascular events
        "cardiovascular_death": Decimal("0.95"),
        "heart_failure_hospitalization": Decimal("0.85"),
        "blood_pressure_reduction": Decimal("0.70"),
        "ldl_reduction": Decimal("0.65"),
        "ef_improvement": Decimal("0.75"),  # Ejection fraction
        "biomarker": Decimal("0.40"),
        "safety": Decimal("0.30"),
    },
    TherapeuticArea.RARE_DISEASE: {
        "functional_improvement": Decimal("1.00"),
        "enzyme_activity": Decimal("0.85"),
        "biomarker_improvement": Decimal("0.80"),
        "quality_of_life": Decimal("0.75"),
        "disease_stabilization": Decimal("0.70"),
        "safety": Decimal("0.40"),
    },
}

# Default weights for unclassified therapeutic areas
DEFAULT_ENDPOINT_WEIGHTS: Dict[str, Decimal] = {
    "overall_survival": Decimal("1.00"),
    "complete_response": Decimal("0.80"),
    "clinical_improvement": Decimal("0.75"),
    "biomarker": Decimal("0.40"),
    "safety": Decimal("0.30"),
    "pk": Decimal("0.20"),
    "tolerability": Decimal("0.20"),
    "dose_finding": Decimal("0.15"),
}


@dataclass
class EndpointWeightConfig:
    """Configuration for endpoint weighting."""
    therapeutic_area: TherapeuticArea
    endpoint_name: str
    base_weight: Decimal
    confidence_adjustment: Decimal = Decimal("1.0")
    is_primary: bool = True


def classify_therapeutic_area(conditions: List[str]) -> TherapeuticArea:
    """
    Classify therapeutic area from trial conditions.

    Args:
        conditions: List of condition strings (lowercased)

    Returns:
        Most likely therapeutic area
    """
    if not conditions:
        return TherapeuticArea.OTHER

    # Combine all conditions into searchable text
    combined = " ".join(conditions).lower()

    # Score each therapeutic area
    area_scores: Dict[TherapeuticArea, int] = {}

    for area, keywords in THERAPEUTIC_AREA_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            area_scores[area] = score

    if not area_scores:
        return TherapeuticArea.OTHER

    # Return highest scoring area (deterministic tie-break by enum order)
    return max(area_scores.keys(), key=lambda a: (area_scores[a], a.value))


def compute_weighted_endpoint_score(
    endpoint_text: str,
    therapeutic_area: TherapeuticArea,
    is_primary_endpoint: bool = True,
) -> Tuple[Decimal, str, bool]:
    """
    Compute weighted endpoint score based on therapeutic area.

    Args:
        endpoint_text: Primary endpoint text
        therapeutic_area: Classified therapeutic area
        is_primary_endpoint: Whether this is a primary endpoint

    Returns:
        (weight, matched_endpoint_type, is_strong)
    """
    if not endpoint_text:
        return (Decimal("0.50"), "unknown", False)

    # Normalize: lowercase and replace hyphens with spaces for matching
    endpoint_lower = endpoint_text.lower().replace("-", " ")

    # Get weights for this therapeutic area
    weights = ENDPOINT_WEIGHTS_BY_AREA.get(
        therapeutic_area,
        DEFAULT_ENDPOINT_WEIGHTS
    )

    # Find best matching endpoint
    best_match = None
    best_weight = Decimal("0.50")  # Default neutral

    for endpoint_type, weight in weights.items():
        # Check for match (word boundary aware)
        pattern = r'\b' + re.escape(endpoint_type.replace("_", " ")) + r'\b'
        if re.search(pattern, endpoint_lower, re.IGNORECASE):
            if weight > best_weight:
                best_weight = weight
                best_match = endpoint_type

        # Also check underscore version
        if endpoint_type in endpoint_lower:
            if weight > best_weight:
                best_weight = weight
                best_match = endpoint_type

    # Apply primary endpoint bonus
    if is_primary_endpoint:
        best_weight = best_weight * Decimal("1.1")
        best_weight = min(best_weight, Decimal("1.0"))

    is_strong = best_weight >= Decimal("0.70")

    return (
        best_weight.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        best_match or "unclassified",
        is_strong,
    )


# =============================================================================
# 2. PHASE-DEPENDENT STALENESS THRESHOLDS
# =============================================================================

@dataclass
class PhaseStalenessConfig:
    """Configuration for phase-dependent staleness."""
    phase: str
    max_staleness_days: int
    warning_threshold_days: int
    severity_if_stale: str  # "sev1", "sev2", "sev3"


# Phase-dependent staleness thresholds
# Later phases should have more recent updates
PHASE_STALENESS_THRESHOLDS: Dict[str, PhaseStalenessConfig] = {
    "approved": PhaseStalenessConfig(
        phase="approved",
        max_staleness_days=365,  # 1 year for commercial
        warning_threshold_days=270,
        severity_if_stale="sev1",
    ),
    "phase 3": PhaseStalenessConfig(
        phase="phase 3",
        max_staleness_days=180,  # 6 months - late stage needs updates
        warning_threshold_days=120,
        severity_if_stale="sev2",
    ),
    "phase 2/3": PhaseStalenessConfig(
        phase="phase 2/3",
        max_staleness_days=240,  # 8 months
        warning_threshold_days=180,
        severity_if_stale="sev2",
    ),
    "phase 2": PhaseStalenessConfig(
        phase="phase 2",
        max_staleness_days=365,  # 1 year
        warning_threshold_days=270,
        severity_if_stale="sev1",
    ),
    "phase 1/2": PhaseStalenessConfig(
        phase="phase 1/2",
        max_staleness_days=540,  # 18 months
        warning_threshold_days=365,
        severity_if_stale="sev1",
    ),
    "phase 1": PhaseStalenessConfig(
        phase="phase 1",
        max_staleness_days=730,  # 2 years
        warning_threshold_days=545,
        severity_if_stale="none",
    ),
    "preclinical": PhaseStalenessConfig(
        phase="preclinical",
        max_staleness_days=1095,  # 3 years
        warning_threshold_days=730,
        severity_if_stale="none",
    ),
    "unknown": PhaseStalenessConfig(
        phase="unknown",
        max_staleness_days=730,  # Default 2 years
        warning_threshold_days=545,
        severity_if_stale="none",
    ),
}


def get_staleness_threshold_for_phase(phase: str) -> PhaseStalenessConfig:
    """Get staleness configuration for a given phase."""
    return PHASE_STALENESS_THRESHOLDS.get(
        phase.lower(),
        PHASE_STALENESS_THRESHOLDS["unknown"]
    )


@dataclass
class PhaseStalenessResult:
    """Result of phase-dependent staleness check."""
    is_stale: bool
    is_warning: bool
    days_since_update: int
    max_allowed_days: int
    warning_threshold_days: int
    phase: str
    severity_applied: str
    staleness_penalty: Decimal


def compute_phase_staleness(
    phase: str,
    last_update_date: Optional[str],
    as_of_date: Union[str, date],
) -> PhaseStalenessResult:
    """
    Compute phase-dependent staleness with appropriate severity.

    Args:
        phase: Clinical trial phase
        last_update_date: Last update date (ISO format)
        as_of_date: Analysis date

    Returns:
        PhaseStalenessResult with staleness assessment
    """
    config = get_staleness_threshold_for_phase(phase)

    # Parse dates
    if isinstance(as_of_date, str):
        as_of_dt = date.fromisoformat(as_of_date)
    else:
        as_of_dt = as_of_date

    if not last_update_date:
        # Unknown update date - assume moderately stale
        return PhaseStalenessResult(
            is_stale=False,
            is_warning=True,
            days_since_update=-1,
            max_allowed_days=config.max_staleness_days,
            warning_threshold_days=config.warning_threshold_days,
            phase=phase,
            severity_applied="none",
            staleness_penalty=Decimal("0.90"),  # 10% penalty for unknown
        )

    try:
        update_dt = date.fromisoformat(last_update_date[:10])
    except (ValueError, TypeError):
        return PhaseStalenessResult(
            is_stale=False,
            is_warning=True,
            days_since_update=-1,
            max_allowed_days=config.max_staleness_days,
            warning_threshold_days=config.warning_threshold_days,
            phase=phase,
            severity_applied="none",
            staleness_penalty=Decimal("0.90"),
        )

    days_since = (as_of_dt - update_dt).days

    if days_since < 0:
        # Future date - data quality issue
        return PhaseStalenessResult(
            is_stale=False,
            is_warning=True,
            days_since_update=days_since,
            max_allowed_days=config.max_staleness_days,
            warning_threshold_days=config.warning_threshold_days,
            phase=phase,
            severity_applied="none",
            staleness_penalty=Decimal("1.0"),
        )

    is_stale = days_since > config.max_staleness_days
    is_warning = days_since > config.warning_threshold_days and not is_stale

    # Calculate penalty
    if is_stale:
        severity = config.severity_if_stale
        if severity == "sev3":
            penalty = Decimal("0.0")  # Complete penalty
        elif severity == "sev2":
            penalty = Decimal("0.50")  # 50% penalty
        elif severity == "sev1":
            penalty = Decimal("0.90")  # 10% penalty
        else:
            penalty = Decimal("0.95")  # Minor penalty
    elif is_warning:
        # Linear interpolation for warning zone
        warning_pct = (days_since - config.warning_threshold_days) / (
            config.max_staleness_days - config.warning_threshold_days
        )
        penalty = Decimal("1.0") - (Decimal(str(warning_pct)) * Decimal("0.10"))
        penalty = max(Decimal("0.90"), penalty)
        severity = "none"
    else:
        penalty = Decimal("1.0")
        severity = "none"

    return PhaseStalenessResult(
        is_stale=is_stale,
        is_warning=is_warning,
        days_since_update=days_since,
        max_allowed_days=config.max_staleness_days,
        warning_threshold_days=config.warning_threshold_days,
        phase=phase,
        severity_applied=severity,
        staleness_penalty=penalty.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
    )


# =============================================================================
# 3. REGULATORY PATHWAY SCORING
# =============================================================================

class RegulatoryDesignation(str, Enum):
    """FDA regulatory designations."""
    BREAKTHROUGH_THERAPY = "breakthrough_therapy"
    FAST_TRACK = "fast_track"
    ACCELERATED_APPROVAL = "accelerated_approval"
    PRIORITY_REVIEW = "priority_review"
    ORPHAN_DRUG = "orphan_drug"
    REMS_REQUIRED = "rems_required"
    STANDARD_PATHWAY = "standard_pathway"


# Regulatory pathway score modifiers
REGULATORY_SCORE_MODIFIERS: Dict[RegulatoryDesignation, Decimal] = {
    RegulatoryDesignation.BREAKTHROUGH_THERAPY: Decimal("15"),    # 40%+ higher approval rate
    RegulatoryDesignation.FAST_TRACK: Decimal("8"),               # Expedited review
    RegulatoryDesignation.ACCELERATED_APPROVAL: Decimal("5"),     # Faster but confirmatory risk
    RegulatoryDesignation.PRIORITY_REVIEW: Decimal("6"),          # 6-month vs 10-month review
    RegulatoryDesignation.ORPHAN_DRUG: Decimal("5"),              # Market exclusivity
    RegulatoryDesignation.REMS_REQUIRED: Decimal("-5"),           # Distribution complexity
    RegulatoryDesignation.STANDARD_PATHWAY: Decimal("0"),         # Baseline
}

# Designation keywords for detection
DESIGNATION_KEYWORDS: Dict[RegulatoryDesignation, List[str]] = {
    RegulatoryDesignation.BREAKTHROUGH_THERAPY: [
        "breakthrough therapy", "breakthrough designation", "btd",
    ],
    RegulatoryDesignation.FAST_TRACK: [
        "fast track", "fast-track", "fasttrack",
    ],
    RegulatoryDesignation.ACCELERATED_APPROVAL: [
        "accelerated approval", "accelerated pathway",
    ],
    RegulatoryDesignation.PRIORITY_REVIEW: [
        "priority review", "priority designation",
    ],
    RegulatoryDesignation.ORPHAN_DRUG: [
        "orphan drug", "orphan designation", "orphan status",
    ],
    RegulatoryDesignation.REMS_REQUIRED: [
        "rems", "risk evaluation", "risk management strategy",
    ],
}


@dataclass
class RegulatoryPathwayScore:
    """Result of regulatory pathway scoring."""
    designations_detected: List[RegulatoryDesignation]
    total_score_modifier: Decimal
    is_expedited: bool  # Any expedited pathway
    has_risk_factor: bool  # REMS or other risk factors
    designation_details: Dict[str, Decimal]
    confidence: str  # "high", "medium", "low"


def compute_regulatory_pathway_score(
    trial_data: Dict[str, Any],
    company_announcements: Optional[List[str]] = None,
) -> RegulatoryPathwayScore:
    """
    Compute regulatory pathway score from trial data and announcements.

    Args:
        trial_data: Trial record with potential designation fields
        company_announcements: Optional list of announcement texts

    Returns:
        RegulatoryPathwayScore with designations and modifiers
    """
    detected: List[RegulatoryDesignation] = []
    designation_details: Dict[str, Decimal] = {}

    # Check explicit designation fields
    designation_fields = [
        "breakthrough_designation",
        "fast_track_designation",
        "accelerated_approval",
        "priority_review",
        "orphan_designation",
        "rems_required",
    ]

    field_to_designation = {
        "breakthrough_designation": RegulatoryDesignation.BREAKTHROUGH_THERAPY,
        "fast_track_designation": RegulatoryDesignation.FAST_TRACK,
        "accelerated_approval": RegulatoryDesignation.ACCELERATED_APPROVAL,
        "priority_review": RegulatoryDesignation.PRIORITY_REVIEW,
        "orphan_designation": RegulatoryDesignation.ORPHAN_DRUG,
        "rems_required": RegulatoryDesignation.REMS_REQUIRED,
    }

    for field_name in designation_fields:
        value = trial_data.get(field_name)
        if value in (True, "true", "yes", "1", 1):
            designation = field_to_designation.get(field_name)
            if designation and designation not in detected:
                detected.append(designation)
                designation_details[designation.value] = REGULATORY_SCORE_MODIFIERS[designation]

    # Check text fields for keyword matches
    text_fields = ["brief_title", "official_title", "description", "conditions"]
    combined_text = ""
    for field in text_fields:
        value = trial_data.get(field, "")
        if isinstance(value, str):
            combined_text += " " + value.lower()
        elif isinstance(value, list):
            combined_text += " " + " ".join(str(v).lower() for v in value)

    # Add announcements
    if company_announcements:
        combined_text += " " + " ".join(a.lower() for a in company_announcements)

    # Keyword detection
    for designation, keywords in DESIGNATION_KEYWORDS.items():
        if designation in detected:
            continue
        for keyword in keywords:
            if keyword in combined_text:
                detected.append(designation)
                designation_details[designation.value] = REGULATORY_SCORE_MODIFIERS[designation]
                break

    # Calculate total modifier (start with Decimal to ensure type consistency)
    total_modifier = sum(
        (REGULATORY_SCORE_MODIFIERS[d] for d in detected),
        Decimal("0")
    )

    # Cap total modifier to prevent extreme scores
    total_modifier = max(Decimal("-10"), min(Decimal("25"), total_modifier))

    # Determine expedited status
    expedited_designations = {
        RegulatoryDesignation.BREAKTHROUGH_THERAPY,
        RegulatoryDesignation.FAST_TRACK,
        RegulatoryDesignation.ACCELERATED_APPROVAL,
        RegulatoryDesignation.PRIORITY_REVIEW,
    }
    is_expedited = bool(set(detected) & expedited_designations)

    # Check for risk factors
    has_risk_factor = RegulatoryDesignation.REMS_REQUIRED in detected

    # Confidence based on detection method
    confidence = "high" if any(
        trial_data.get(f) for f in designation_fields
    ) else "medium" if detected else "low"

    return RegulatoryPathwayScore(
        designations_detected=detected,
        total_score_modifier=total_modifier.quantize(Decimal("0.01")),
        is_expedited=is_expedited,
        has_risk_factor=has_risk_factor,
        designation_details=designation_details,
        confidence=confidence,
    )


# =============================================================================
# 4. REGIME-ADAPTIVE CATALYST DECAY
# =============================================================================

class MarketRegimeType(str, Enum):
    """Market regime classifications for decay adjustment."""
    BULL = "bull"
    BEAR = "bear"
    VOLATILITY_SPIKE = "volatility_spike"
    SECTOR_ROTATION = "sector_rotation"
    UNKNOWN = "unknown"


@dataclass
class RegimeDecayConfig:
    """Configuration for regime-adaptive decay."""
    regime: MarketRegimeType
    decay_half_life_days: int
    rationale: str


# Regime-specific decay constants
# Bull: Market digests news quickly (20 days)
# Bear/Volatility: Delayed reactions, institutional windows (45 days)
# Sector rotation: Normal (30 days)
REGIME_DECAY_CONFIGS: Dict[MarketRegimeType, RegimeDecayConfig] = {
    MarketRegimeType.BULL: RegimeDecayConfig(
        regime=MarketRegimeType.BULL,
        decay_half_life_days=20,
        rationale="Risk-on markets digest catalyst news quickly",
    ),
    MarketRegimeType.BEAR: RegimeDecayConfig(
        regime=MarketRegimeType.BEAR,
        decay_half_life_days=45,
        rationale="Delayed institutional reactions in risk-off environment",
    ),
    MarketRegimeType.VOLATILITY_SPIKE: RegimeDecayConfig(
        regime=MarketRegimeType.VOLATILITY_SPIKE,
        decay_half_life_days=45,
        rationale="High uncertainty extends catalyst relevance window",
    ),
    MarketRegimeType.SECTOR_ROTATION: RegimeDecayConfig(
        regime=MarketRegimeType.SECTOR_ROTATION,
        decay_half_life_days=30,
        rationale="Normal market conditions, standard decay",
    ),
    MarketRegimeType.UNKNOWN: RegimeDecayConfig(
        regime=MarketRegimeType.UNKNOWN,
        decay_half_life_days=30,
        rationale="Default decay when regime unknown",
    ),
}


@dataclass
class RegimeDecayResult:
    """Result of regime-adaptive decay calculation."""
    decay_half_life_days: int
    decay_weight: Decimal
    regime: MarketRegimeType
    days_since_event: int
    rationale: str


def compute_regime_adaptive_decay(
    event_date: str,
    as_of_date: Union[str, date],
    regime: MarketRegimeType,
) -> RegimeDecayResult:
    """
    Compute regime-adaptive recency decay weight.

    Args:
        event_date: Event date (ISO format)
        as_of_date: Analysis date
        regime: Current market regime

    Returns:
        RegimeDecayResult with adaptive decay weight
    """
    config = REGIME_DECAY_CONFIGS.get(regime, REGIME_DECAY_CONFIGS[MarketRegimeType.UNKNOWN])

    # Parse dates
    if isinstance(as_of_date, str):
        as_of_dt = date.fromisoformat(as_of_date)
    else:
        as_of_dt = as_of_date

    try:
        event_dt = date.fromisoformat(event_date[:10])
    except (ValueError, TypeError):
        # Invalid date - return neutral weight
        return RegimeDecayResult(
            decay_half_life_days=config.decay_half_life_days,
            decay_weight=Decimal("0.5"),
            regime=regime,
            days_since_event=-1,
            rationale="Invalid event date",
        )

    days_since = (as_of_dt - event_dt).days

    if days_since < 0:
        # Future event
        return RegimeDecayResult(
            decay_half_life_days=config.decay_half_life_days,
            decay_weight=Decimal("0"),
            regime=regime,
            days_since_event=days_since,
            rationale="Future event - not scored",
        )

    if days_since == 0:
        weight = Decimal("1.0")
    else:
        # Exponential decay: weight = 2^(-days / half_life)
        decay_factor = Decimal(str(days_since)) / Decimal(str(config.decay_half_life_days))
        weight = Decimal("0.5") ** decay_factor
        weight = weight.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    return RegimeDecayResult(
        decay_half_life_days=config.decay_half_life_days,
        decay_weight=weight,
        regime=regime,
        days_since_event=days_since,
        rationale=config.rationale,
    )


# =============================================================================
# 5. COMPETITIVE LANDSCAPE PENALTY
# =============================================================================

@dataclass
class CompetitiveLandscapeResult:
    """Result of competitive landscape analysis."""
    indication: str
    phase: str
    competitor_count: int
    competition_level: str  # "low", "moderate", "high", "hyper_competitive"
    penalty_points: Decimal
    market_share_estimate: Decimal  # Estimated share if approved
    is_first_in_class: bool


# Competition thresholds and penalties
COMPETITION_THRESHOLDS = {
    "low": (0, 2, Decimal("0")),           # 0-2 competitors: no penalty
    "moderate": (3, 5, Decimal("5")),       # 3-5 competitors: 5 pt penalty
    "high": (6, 10, Decimal("12")),         # 6-10 competitors: 12 pt penalty
    "hyper_competitive": (11, 999, Decimal("20")),  # 11+ competitors: 20 pt penalty
}


def compute_competition_penalty(
    indication: str,
    phase: str,
    competitor_programs: List[Dict[str, Any]],
    is_first_in_class: bool = False,
) -> CompetitiveLandscapeResult:
    """
    Compute competitive landscape penalty.

    Args:
        indication: Target indication
        phase: Trial phase
        competitor_programs: List of competitor program records
        is_first_in_class: Whether this is a novel mechanism

    Returns:
        CompetitiveLandscapeResult with penalty assessment
    """
    # Count competitors at same or later phase
    phase_order = {
        "preclinical": 0,
        "phase 1": 1,
        "phase 1/2": 2,
        "phase 2": 3,
        "phase 2/3": 4,
        "phase 3": 5,
        "approved": 6,
    }

    current_phase_level = phase_order.get(phase.lower(), 3)

    competitor_count = 0
    for comp in competitor_programs:
        comp_phase = comp.get("phase", "").lower()
        comp_phase_level = phase_order.get(comp_phase, 3)
        if comp_phase_level >= current_phase_level:
            competitor_count += 1

    # Determine competition level
    competition_level = "low"
    penalty = Decimal("0")

    for level, (min_count, max_count, level_penalty) in COMPETITION_THRESHOLDS.items():
        if min_count <= competitor_count <= max_count:
            competition_level = level
            penalty = level_penalty
            break

    # First-in-class bonus: reduce penalty by 50%
    if is_first_in_class:
        penalty = penalty * Decimal("0.5")

    # Phase adjustment: earlier phases get reduced penalty (more can drop out)
    if current_phase_level <= 2:  # Phase 1 or earlier
        penalty = penalty * Decimal("0.5")
    elif current_phase_level <= 4:  # Phase 2
        penalty = penalty * Decimal("0.75")

    # Estimate market share
    total_programs = competitor_count + 1
    market_share = Decimal("1") / Decimal(str(max(total_programs, 1)))
    market_share = market_share.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return CompetitiveLandscapeResult(
        indication=indication,
        phase=phase,
        competitor_count=competitor_count,
        competition_level=competition_level,
        penalty_points=penalty.quantize(Decimal("0.01")),
        market_share_estimate=market_share,
        is_first_in_class=is_first_in_class,
    )


# =============================================================================
# 6. DYNAMIC DILUTION (VIX-ADJUSTED)
# =============================================================================

@dataclass
class VixDilutionAdjustment:
    """Result of VIX-adjusted dilution calculation."""
    vix_level: Decimal
    vix_bucket: str  # "low", "normal", "elevated", "high", "extreme"
    adjustment_factor: Decimal  # Multiplier on dilution risk
    cost_of_equity_premium: Decimal  # Estimated premium over risk-free
    market_receptiveness: str  # "favorable", "neutral", "challenging", "difficult"


# VIX thresholds and dilution risk adjustments
VIX_DILUTION_THRESHOLDS: Dict[str, Tuple[Decimal, Decimal, Decimal, str]] = {
    # bucket: (min_vix, max_vix, adjustment_factor, market_receptiveness)
    "low": (Decimal("0"), Decimal("15"), Decimal("0.85"), "favorable"),
    "normal": (Decimal("15"), Decimal("20"), Decimal("1.00"), "neutral"),
    "elevated": (Decimal("20"), Decimal("25"), Decimal("1.15"), "challenging"),
    "high": (Decimal("25"), Decimal("35"), Decimal("1.35"), "difficult"),
    "extreme": (Decimal("35"), Decimal("100"), Decimal("1.60"), "very_difficult"),
}


def compute_vix_dilution_adjustment(
    vix_current: Decimal,
    base_dilution_score: Decimal,
) -> VixDilutionAdjustment:
    """
    Compute VIX-adjusted dilution risk.

    Higher VIX = harder to raise equity = higher dilution risk.

    Args:
        vix_current: Current VIX level
        base_dilution_score: Base dilution score (0-100)

    Returns:
        VixDilutionAdjustment with adjusted risk assessment
    """
    # Determine VIX bucket
    vix_bucket = "normal"
    adjustment_factor = Decimal("1.0")
    market_receptiveness = "neutral"

    for bucket, (min_vix, max_vix, factor, receptiveness) in VIX_DILUTION_THRESHOLDS.items():
        if min_vix <= vix_current < max_vix:
            vix_bucket = bucket
            adjustment_factor = factor
            market_receptiveness = receptiveness
            break

    # Extreme VIX catch-all
    if vix_current >= Decimal("35"):
        vix_bucket = "extreme"
        adjustment_factor = Decimal("1.60")
        market_receptiveness = "very_difficult"

    # Estimate cost of equity premium (rough approximation)
    # Base premium ~5%, increases with VIX
    base_premium = Decimal("5.0")
    vix_premium = (vix_current - Decimal("15")) * Decimal("0.3")
    vix_premium = max(Decimal("0"), vix_premium)
    cost_of_equity_premium = base_premium + vix_premium
    cost_of_equity_premium = cost_of_equity_premium.quantize(Decimal("0.1"))

    return VixDilutionAdjustment(
        vix_level=vix_current,
        vix_bucket=vix_bucket,
        adjustment_factor=adjustment_factor.quantize(Decimal("0.01")),
        cost_of_equity_premium=cost_of_equity_premium,
        market_receptiveness=market_receptiveness,
    )


# =============================================================================
# 7. BURN SEASONALITY ADJUSTMENT
# =============================================================================

@dataclass
class QuarterlySeasonalityConfig:
    """Configuration for quarterly burn seasonality."""
    quarter: int  # 1-4
    adjustment_factor: Decimal
    rationale: str


# Quarterly burn seasonality factors
# Q1: Lower (post-JPM financing, planning)
# Q2-Q3: Higher (enrollment ramps, clinical activity)
# Q4: Highest (R&D milestones, year-end submissions)
QUARTERLY_BURN_SEASONALITY: Dict[int, QuarterlySeasonalityConfig] = {
    1: QuarterlySeasonalityConfig(
        quarter=1,
        adjustment_factor=Decimal("0.85"),
        rationale="Post-JPM financing, planning period, lower R&D activity",
    ),
    2: QuarterlySeasonalityConfig(
        quarter=2,
        adjustment_factor=Decimal("1.05"),
        rationale="Enrollment ramps up, increased clinical activity",
    ),
    3: QuarterlySeasonalityConfig(
        quarter=3,
        adjustment_factor=Decimal("1.10"),
        rationale="Peak enrollment, manufacturing prep, site costs",
    ),
    4: QuarterlySeasonalityConfig(
        quarter=4,
        adjustment_factor=Decimal("1.00"),
        rationale="Normalized baseline (historically high but expected)",
    ),
}


@dataclass
class BurnSeasonalityResult:
    """Result of burn seasonality adjustment."""
    fiscal_quarter: int
    adjustment_factor: Decimal
    adjusted_monthly_burn: Decimal
    adjusted_runway_months: Decimal
    rationale: str
    is_q4_submission_window: bool


def compute_burn_seasonality_adjustment(
    monthly_burn: Decimal,
    liquid_assets: Decimal,
    as_of_date: Union[str, date],
    fiscal_year_end_month: int = 12,  # December default
) -> BurnSeasonalityResult:
    """
    Compute seasonality-adjusted burn rate and runway.

    Args:
        monthly_burn: Base monthly burn rate
        liquid_assets: Current liquid assets
        as_of_date: Analysis date
        fiscal_year_end_month: Fiscal year end month (1-12)

    Returns:
        BurnSeasonalityResult with adjusted burn and runway
    """
    # Parse date
    if isinstance(as_of_date, str):
        as_of_dt = date.fromisoformat(as_of_date)
    else:
        as_of_dt = as_of_date

    # Determine fiscal quarter
    month = as_of_dt.month
    fiscal_month = ((month - fiscal_year_end_month - 1) % 12) + 1

    if fiscal_month <= 3:
        fiscal_quarter = 1
    elif fiscal_month <= 6:
        fiscal_quarter = 2
    elif fiscal_month <= 9:
        fiscal_quarter = 3
    else:
        fiscal_quarter = 4

    # Get seasonality config
    config = QUARTERLY_BURN_SEASONALITY.get(
        fiscal_quarter,
        QUARTERLY_BURN_SEASONALITY[4]
    )

    # Apply adjustment
    adjusted_burn = monthly_burn * config.adjustment_factor
    adjusted_burn = adjusted_burn.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Calculate adjusted runway
    if adjusted_burn > Decimal("0"):
        adjusted_runway = liquid_assets / adjusted_burn
    else:
        adjusted_runway = Decimal("999")

    adjusted_runway = adjusted_runway.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    # Check if in Q4 FDA submission window (Oct-Dec for calendar year)
    is_q4_submission = fiscal_quarter == 4

    return BurnSeasonalityResult(
        fiscal_quarter=fiscal_quarter,
        adjustment_factor=config.adjustment_factor,
        adjusted_monthly_burn=adjusted_burn,
        adjusted_runway_months=adjusted_runway,
        rationale=config.rationale,
        is_q4_submission_window=is_q4_submission,
    )


# =============================================================================
# 8. BINARY EVENT PROXIMITY BOOST
# =============================================================================

@dataclass
class ProximityBoostResult:
    """Result of binary event proximity boost calculation."""
    days_to_event: int
    proximity_bucket: str  # "imminent", "near_term", "medium_term", "far", "none"
    boost_percentage: Decimal
    catalyst_type: str
    confidence: str
    boosted_catalyst_weight: Decimal


# Proximity boost thresholds
# 0-30 days: +20% catalyst weight
# 31-60 days: +10% catalyst weight
# 61-90 days: +5% catalyst weight
PROXIMITY_BOOST_THRESHOLDS: Dict[str, Tuple[int, int, Decimal]] = {
    "imminent": (0, 30, Decimal("0.20")),      # +20%
    "near_term": (31, 60, Decimal("0.10")),    # +10%
    "medium_term": (61, 90, Decimal("0.05")),  # +5%
    "far": (91, 999, Decimal("0.00")),         # No boost
}


def compute_binary_event_proximity_boost(
    next_catalyst_date: Optional[str],
    as_of_date: Union[str, date],
    catalyst_type: str = "unknown",
    confidence: str = "medium",
    base_catalyst_weight: Decimal = Decimal("0.25"),
) -> ProximityBoostResult:
    """
    Compute proximity boost for upcoming binary events.

    Args:
        next_catalyst_date: Date of next catalyst (ISO format)
        as_of_date: Analysis date
        catalyst_type: Type of catalyst event
        confidence: Confidence level (high, medium, low)
        base_catalyst_weight: Base weight for catalyst in composite

    Returns:
        ProximityBoostResult with boost assessment
    """
    if not next_catalyst_date:
        return ProximityBoostResult(
            days_to_event=-1,
            proximity_bucket="none",
            boost_percentage=Decimal("0"),
            catalyst_type=catalyst_type,
            confidence=confidence,
            boosted_catalyst_weight=base_catalyst_weight,
        )

    # Parse dates
    if isinstance(as_of_date, str):
        as_of_dt = date.fromisoformat(as_of_date)
    else:
        as_of_dt = as_of_date

    try:
        catalyst_dt = date.fromisoformat(next_catalyst_date[:10])
    except (ValueError, TypeError):
        return ProximityBoostResult(
            days_to_event=-1,
            proximity_bucket="none",
            boost_percentage=Decimal("0"),
            catalyst_type=catalyst_type,
            confidence=confidence,
            boosted_catalyst_weight=base_catalyst_weight,
        )

    days_to_event = (catalyst_dt - as_of_dt).days

    if days_to_event < 0:
        # Past event
        return ProximityBoostResult(
            days_to_event=days_to_event,
            proximity_bucket="none",
            boost_percentage=Decimal("0"),
            catalyst_type=catalyst_type,
            confidence=confidence,
            boosted_catalyst_weight=base_catalyst_weight,
        )

    # Determine proximity bucket and boost
    proximity_bucket = "far"
    boost_percentage = Decimal("0")

    for bucket, (min_days, max_days, boost) in PROXIMITY_BOOST_THRESHOLDS.items():
        if min_days <= days_to_event <= max_days:
            proximity_bucket = bucket
            boost_percentage = boost
            break

    # Confidence adjustment: reduce boost for low confidence
    confidence_multipliers = {
        "high": Decimal("1.0"),
        "medium": Decimal("0.75"),
        "low": Decimal("0.5"),
    }
    confidence_mult = confidence_multipliers.get(confidence.lower(), Decimal("0.75"))
    adjusted_boost = boost_percentage * confidence_mult

    # Calculate boosted weight
    boosted_weight = base_catalyst_weight * (Decimal("1") + adjusted_boost)
    boosted_weight = boosted_weight.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    return ProximityBoostResult(
        days_to_event=days_to_event,
        proximity_bucket=proximity_bucket,
        boost_percentage=adjusted_boost.quantize(Decimal("0.001")),
        catalyst_type=catalyst_type,
        confidence=confidence,
        boosted_catalyst_weight=boosted_weight,
    )


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def apply_all_accuracy_improvements(
    ticker: str,
    trial_data: Dict[str, Any],
    financial_data: Dict[str, Any],
    market_data: Dict[str, Any],
    as_of_date: Union[str, date],
    vix_current: Optional[Decimal] = None,
    market_regime: Optional[MarketRegimeType] = None,
    competitor_programs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Apply all accuracy improvements and return combined adjustments.

    This is a convenience function for integrating all improvements.

    Args:
        ticker: Ticker symbol
        trial_data: Clinical trial data
        financial_data: Financial data
        market_data: Market data
        as_of_date: Analysis date
        vix_current: Current VIX (optional)
        market_regime: Current market regime (optional)
        competitor_programs: Competitor program list (optional)

    Returns:
        Dict with all improvement results
    """
    results = {"ticker": ticker, "as_of_date": str(as_of_date)}

    # 1. Indication-specific endpoint weighting
    conditions = trial_data.get("conditions", [])
    if isinstance(conditions, str):
        conditions = [conditions]
    therapeutic_area = classify_therapeutic_area(conditions)
    endpoint = trial_data.get("primary_endpoint", "")
    endpoint_weight, endpoint_type, is_strong = compute_weighted_endpoint_score(
        endpoint, therapeutic_area
    )
    results["endpoint_analysis"] = {
        "therapeutic_area": therapeutic_area.value,
        "endpoint_weight": str(endpoint_weight),
        "endpoint_type": endpoint_type,
        "is_strong": is_strong,
    }

    # 2. Phase-dependent staleness
    phase = trial_data.get("phase", "unknown")
    last_update = trial_data.get("last_update_posted")
    staleness = compute_phase_staleness(phase, last_update, as_of_date)
    results["staleness_analysis"] = {
        "is_stale": staleness.is_stale,
        "is_warning": staleness.is_warning,
        "days_since_update": staleness.days_since_update,
        "staleness_penalty": str(staleness.staleness_penalty),
        "severity_applied": staleness.severity_applied,
    }

    # 3. Regulatory pathway scoring
    regulatory = compute_regulatory_pathway_score(trial_data)
    results["regulatory_analysis"] = {
        "designations": [d.value for d in regulatory.designations_detected],
        "total_modifier": str(regulatory.total_score_modifier),
        "is_expedited": regulatory.is_expedited,
        "has_risk_factor": regulatory.has_risk_factor,
    }

    # 4. Regime-adaptive catalyst decay (if event date available)
    event_date = trial_data.get("event_date") or trial_data.get("last_update_posted")
    if event_date and market_regime:
        decay = compute_regime_adaptive_decay(event_date, as_of_date, market_regime)
        results["decay_analysis"] = {
            "decay_half_life_days": decay.decay_half_life_days,
            "decay_weight": str(decay.decay_weight),
            "regime": decay.regime.value,
        }

    # 5. Competitive landscape penalty (if competitors provided)
    indication = conditions[0] if conditions else "unknown"
    if competitor_programs is not None:
        competition = compute_competition_penalty(
            indication, phase, competitor_programs
        )
        results["competition_analysis"] = {
            "competitor_count": competition.competitor_count,
            "competition_level": competition.competition_level,
            "penalty_points": str(competition.penalty_points),
            "market_share_estimate": str(competition.market_share_estimate),
        }

    # 6. VIX-adjusted dilution (if VIX provided)
    dilution_score = financial_data.get("dilution_score", Decimal("50"))
    if isinstance(dilution_score, (int, float)):
        dilution_score = Decimal(str(dilution_score))
    if vix_current:
        vix_adj = compute_vix_dilution_adjustment(vix_current, dilution_score)
        results["vix_adjustment"] = {
            "vix_bucket": vix_adj.vix_bucket,
            "adjustment_factor": str(vix_adj.adjustment_factor),
            "market_receptiveness": vix_adj.market_receptiveness,
            "cost_of_equity_premium": str(vix_adj.cost_of_equity_premium),
        }

    # 7. Burn seasonality
    monthly_burn = financial_data.get("monthly_burn", Decimal("0"))
    liquid_assets = financial_data.get("liquid_assets", Decimal("0"))
    if isinstance(monthly_burn, (int, float)):
        monthly_burn = Decimal(str(monthly_burn))
    if isinstance(liquid_assets, (int, float)):
        liquid_assets = Decimal(str(liquid_assets))
    if monthly_burn > 0:
        seasonality = compute_burn_seasonality_adjustment(
            monthly_burn, liquid_assets, as_of_date
        )
        results["seasonality_analysis"] = {
            "fiscal_quarter": seasonality.fiscal_quarter,
            "adjustment_factor": str(seasonality.adjustment_factor),
            "adjusted_monthly_burn": str(seasonality.adjusted_monthly_burn),
            "adjusted_runway_months": str(seasonality.adjusted_runway_months),
        }

    # 8. Binary event proximity boost
    next_catalyst = trial_data.get("next_catalyst_date")
    if next_catalyst:
        proximity = compute_binary_event_proximity_boost(
            next_catalyst, as_of_date,
            catalyst_type=trial_data.get("catalyst_type", "unknown"),
            confidence=trial_data.get("catalyst_confidence", "medium"),
        )
        results["proximity_analysis"] = {
            "days_to_event": proximity.days_to_event,
            "proximity_bucket": proximity.proximity_bucket,
            "boost_percentage": str(proximity.boost_percentage),
            "boosted_catalyst_weight": str(proximity.boosted_catalyst_weight),
        }

    return results


# =============================================================================
# SELF-CHECKS
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify correctness."""
    errors = []

    # CHECK 1: Therapeutic area classification
    oncology_conditions = ["breast cancer", "nsclc"]
    area = classify_therapeutic_area(oncology_conditions)
    if area != TherapeuticArea.ONCOLOGY:
        errors.append(f"CHECK1 FAIL: Expected ONCOLOGY, got {area}")

    # CHECK 2: Endpoint weighting
    weight, etype, is_strong = compute_weighted_endpoint_score(
        "Overall Survival", TherapeuticArea.ONCOLOGY
    )
    if weight < Decimal("0.95") or not is_strong:
        errors.append(f"CHECK2 FAIL: OS weight {weight}, is_strong {is_strong}")

    # CHECK 3: Phase staleness
    result = compute_phase_staleness("phase 3", "2024-01-01", "2026-01-15")
    if not result.is_stale:
        errors.append(f"CHECK3 FAIL: Phase 3 from 2024-01 should be stale by 2026-01")

    # CHECK 4: Regulatory scoring
    trial_data = {"breakthrough_designation": True}
    reg_score = compute_regulatory_pathway_score(trial_data)
    if RegulatoryDesignation.BREAKTHROUGH_THERAPY not in reg_score.designations_detected:
        errors.append("CHECK4 FAIL: Breakthrough not detected")

    # CHECK 5: Regime decay
    decay = compute_regime_adaptive_decay(
        "2026-01-01", "2026-01-15", MarketRegimeType.BULL
    )
    if decay.decay_half_life_days != 20:
        errors.append(f"CHECK5 FAIL: Bull regime half-life should be 20, got {decay.decay_half_life_days}")

    # CHECK 6: Competition penalty
    comp = compute_competition_penalty(
        "breast cancer", "phase 3",
        [{"phase": "phase 3"} for _ in range(8)]
    )
    if comp.competition_level != "high":
        errors.append(f"CHECK6 FAIL: 8 competitors should be 'high', got {comp.competition_level}")

    # CHECK 7: VIX adjustment
    vix_adj = compute_vix_dilution_adjustment(Decimal("30"), Decimal("50"))
    if vix_adj.adjustment_factor <= Decimal("1.0"):
        errors.append(f"CHECK7 FAIL: VIX 30 should increase dilution risk")

    # CHECK 8: Burn seasonality
    season = compute_burn_seasonality_adjustment(
        Decimal("10000000"), Decimal("100000000"), "2026-01-15"
    )
    if season.adjustment_factor != Decimal("0.85"):
        errors.append(f"CHECK8 FAIL: Q1 adjustment should be 0.85, got {season.adjustment_factor}")

    # CHECK 9: Proximity boost
    prox = compute_binary_event_proximity_boost(
        "2026-02-01", "2026-01-15"  # 17 days away
    )
    if prox.proximity_bucket != "imminent":
        errors.append(f"CHECK9 FAIL: 17 days should be 'imminent', got {prox.proximity_bucket}")

    return errors


if __name__ == "__main__":
    print("Running accuracy improvements self-checks...")
    errors = _run_self_checks()
    if errors:
        print("SELF-CHECK FAILURES:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All self-checks passed!")
