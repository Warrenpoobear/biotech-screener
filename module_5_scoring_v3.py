#!/usr/bin/env python3
"""
Module 5 Scoring: Per-ticker scoring logic extracted from module_5_composite_v3.

This module contains:
- Types: MonotonicCap, ScoringMode, RunStatus, NormalizationMethod, ComponentScore, ScoreBreakdown, V3ScoringResult
- Helper functions: coalesce, bucket classifiers, confidence extractors, normalization
- Main scoring function: _score_single_ticker_v3

Extracted to reduce module_5_composite_v3.py complexity while preserving:
- Deterministic behavior (no logic changes during extraction)
- Import stability (re-exported from composite)
- Clean dependency graph (scoring does not import composite)

Author: Wake Robin Capital Management
Version: 3.0.0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.types import Severity

# Import Financial Module 2: Survivability scoring
from financial_module_2_survivability import compute_survivability_score

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    # Core enhancement functions
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal_with_fallback,
    compute_momentum_signal_multiwindow,
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    detect_contradictions,  # Enhancement 6: Contradiction detector
    # Types
    VolatilityAdjustment,
    VolatilityBucket,
    MomentumSignal,
    MultiWindowMomentumInput,
    ValuationSignal,
    ValuationRegime,
    CatalystDecayResult,
    SmartMoneySignal,
    InteractionTerms,
    ContradictionResult,  # Enhancement 6: Contradiction result type
    # Helpers
    _to_decimal,
    _quantize_score,
    _quantize_weight,
    _clamp,
    EPS,
    SCORE_PRECISION,
)

# =============================================================================
# VERSION CONSTANTS (used by scoring for determinism hash)
# =============================================================================

SCHEMA_VERSION = "v3.0"

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# Severity multipliers
SEVERITY_MULTIPLIERS = {
    Severity.NONE: Decimal("1.0"),
    Severity.SEV1: Decimal("0.90"),  # 10% penalty
    Severity.SEV2: Decimal("0.50"),  # 50% penalty (soft gate)
    Severity.SEV3: Decimal("0.0"),   # Hard gate (excluded)
}

SEVERITY_GATE_LABELS = {
    Severity.NONE: "NONE",
    Severity.SEV1: "SEV1_10PCT",
    Severity.SEV2: "SEV2_HALF",
    Severity.SEV3: "SEV3_EXCLUDE",
}

# Cohort parameters
MIN_COHORT_SIZE = 5
MAX_UNCERTAINTY_PENALTY = Decimal("0.30")

# Winsorization bounds
WINSOR_LOW = Decimal("5")
WINSOR_HIGH = Decimal("95")

# Hybrid aggregation
HYBRID_ALPHA = Decimal("0.85")
CRITICAL_COMPONENTS = ["financial", "clinical"]

# PoS contribution cap
POS_DELTA_CAP = Decimal("6.0")

# Catalyst effective blending weights
CATALYST_WINDOW_WEIGHT = Decimal("0.70")
CATALYST_PROXIMITY_WEIGHT = Decimal("0.30")
CATALYST_DEFAULT_BASE = Decimal("0.85")
CATALYST_DEFAULT_SCORE = Decimal("50")

# Confidence gate threshold
CONFIDENCE_GATE_THRESHOLD = Decimal("0.4")

# =============================================================================
# ENHANCEMENT 1: HARD REGIME GATING CONFIGURATION
# =============================================================================

REGIME_GATE_CONFIG = {
    "BEAR": {
        "momentum_cap_pct": Decimal("0.30"),      # Cap momentum at 30% of deviation from neutral
        "valuation_upside_cap": Decimal("55"),    # Cap valuation score at 55
        "financial_penalty_mult": Decimal("1.25"), # Amplify financial penalties 25%
    },
    "BULL": {
        "momentum_cap_pct": Decimal("1.0"),       # Full momentum
        "catalyst_boost": Decimal("1.15"),        # 15% catalyst boost
        "financial_penalty_mult": Decimal("0.85"), # Soften penalties 15%
    },
    "NEUTRAL": {},  # No gating
}

# =============================================================================
# ENHANCEMENT 2: EXISTENTIAL FLAW CONFIGURATION
# =============================================================================

EXISTENTIAL_FLAW_CONFIG = {
    "runway_critical_months": Decimal("9"),
    "binary_clinical_risk_phases": ["phase_1", "phase_2", "phase 1", "phase 2", "p1", "p2"],
    "existential_cap": Decimal("65"),
    "existential_penalty": Decimal("0.20"),  # 20% penalty alternative
}

# =============================================================================
# ENHANCEMENT 3: CONFIDENCE-WEIGHTED AGGREGATION
# =============================================================================

CONFIDENCE_WEIGHTED_CONFIG = {
    "confidence_floor": Decimal("0.3"),  # Minimum confidence factor applied
}

# =============================================================================
# ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
# =============================================================================

STAGE_CEILING_CONFIG = {
    "preclinical": Decimal("65"),
    "phase_1": Decimal("70"),
    "phase 1": Decimal("70"),
    "p1": Decimal("70"),
    # phase_2+ has no stage ceiling
}

CATALYST_CEILING_CONFIG = {
    "no_catalyst_12mo": Decimal("75"),
    "no_catalyst_6mo": Decimal("70"),
}

COMMERCIAL_CEILING_CONFIG = {
    "revenue_declining": Decimal("70"),  # YoY revenue decline
}

# =============================================================================
# ENHANCEMENT 5: ASYMMETRIC TRANSFORM CONFIGURATION
# =============================================================================

ASYMMETRY_CONFIG = {
    "upside_dampening": Decimal("0.6"),       # +10 → +6
    "downside_amplification": Decimal("1.2"), # -10 → -12
    "neutral_threshold": Decimal("50"),       # Scores above/below this
}

# =============================================================================
# COVERAGE-GATED SMART MONEY WEIGHTING (OPTION A)
# =============================================================================
# If a security has 13F coverage (any elite manager holds it), include smart_money
# at the configured weight. If no coverage, set smart_money weight to 0 and
# renormalize other weights. This prevents penalizing uncovered securities while
# still allowing SM signal to influence rankings for covered names.
#
# Key invariant: Weights always sum to 1.0 after renormalization.

SMART_MONEY_COVERAGE_GATED_WEIGHT = Decimal("0.05")  # 5% when coverage exists

# =============================================================================
# ENHANCEMENT 8: SMART MONEY SIGNAL ENHANCEMENT (A+B+C+E)
# =============================================================================
# A: Stage×size dependent base weights
# B: Expanded score range with tier-1 boosts
# C: Cohort-normalized overlap (z-score within stage×size)
# E: Confidence from overlap count

SMART_MONEY_ENHANCEMENT_CONFIG = {
    "enabled": True,  # Master switch

    # A: Stage×size dependent base weights
    "base_weights_by_stage_size": {
        "early": {
            "micro": Decimal("0.07"), "small": Decimal("0.07"),
            "mid": Decimal("0.06"), "large": Decimal("0.05")
        },
        "poc": {  # PoC = highest predictive power for smart money
            "micro": Decimal("0.10"), "small": Decimal("0.09"),
            "mid": Decimal("0.08"), "large": Decimal("0.07")
        },
        "pivotal": {
            "micro": Decimal("0.08"), "small": Decimal("0.08"),
            "mid": Decimal("0.07"), "large": Decimal("0.06")
        },
        "regulatory": {  # Regulatory = also high predictive power
            "micro": Decimal("0.09"), "small": Decimal("0.09"),
            "mid": Decimal("0.08"), "large": Decimal("0.07")
        },
        "commercial": {
            "micro": Decimal("0.06"), "small": Decimal("0.06"),
            "mid": Decimal("0.05"), "large": Decimal("0.04")
        },
    },

    # B: Expanded score range parameters
    "score_range": {
        "min": Decimal("20"),
        "max": Decimal("85"),
        "neutral": Decimal("50"),
        "per_tier1_holder_boost": Decimal("2.5"),  # Each Tier-1 adds +2.5 points
        "per_position_increase_boost": Decimal("3.0"),  # Each "INCREASE" adds +3
        "max_boost_from_holders": Decimal("15.0"),  # Cap total holder boost
    },

    # C: Cohort normalization parameters
    "cohort_normalization": {
        "enabled": True,
        "min_cohort_size": 5,
        "zscore_to_score_factor": Decimal("15.0"),  # 1σ = 15 points
        "floor_overlap_for_cohort": 2,  # Need ≥2 holders to be in cohort stats
    },

    # E: Confidence from overlap
    "confidence_from_overlap": {
        "base_confidence": Decimal("0.5"),
        "per_overlap_boost": Decimal("0.08"),  # +0.08 per holder
        "per_tier1_boost": Decimal("0.12"),    # +0.12 per Tier-1
        "max_confidence": Decimal("0.95"),
    },
}

# =============================================================================
# ENHANCEMENT 7: STAGE/SIZE AWARE WEIGHTING
# =============================================================================
# Apply multiplicative tilts to base weights based on development stage and
# market cap. This adjusts component importance based on what matters most
# at each stage of a company's lifecycle.
#
# Order of operations: base → stage/size → regime → vol → confidence → smart_money

STAGE_SIZE_TILT_CONFIG = {
    "enabled": True,  # Master switch for tilting

    # Stage tilts (multiplicative) - what matters at each development phase
    "stage_tilts": {
        "early": {  # Preclinical → Phase 1
            "clinical": Decimal("1.35"),    # ↑ Science risk dominant
            "pos": Decimal("1.25"),         # ↑ Binary outcomes
            "financial": Decimal("1.20"),   # ↑ Survivability critical
            "short_interest": Decimal("1.10"),  # ↑ Micro-cap positioning
            "catalyst": Decimal("0.90"),    # → Events far out, low quality
            "valuation": Decimal("0.75"),   # ↓ No fundamentals
            "momentum": Decimal("0.90"),    # ↓ No commercial traction
        },
        "poc": {  # Phase 2 → Proof-of-concept
            "catalyst": Decimal("1.50"),    # ↑↑ Readout events price in
            "clinical": Decimal("1.10"),    # ↑ PoC data matters
            "short_interest": Decimal("1.15"),  # ↑ Positioning around binary
            "momentum": Decimal("1.10"),    # ↑ Pre-readout moves
            "pos": Decimal("1.05"),         # → Refined PoS post-data
            "financial": Decimal("0.95"),   # → Less critical pre-commercial
            "valuation": Decimal("0.85"),   # → Still speculative
        },
        "pivotal": {  # Phase 3 → Regulatory submission
            "catalyst": Decimal("1.25"),    # ↑ Pivotal data/regulatory events
            "financial": Decimal("1.10"),   # ↑ Capital needs for launch
            "clinical": Decimal("1.00"),    # → Mature dataset
            "valuation": Decimal("1.10"),   # ↑ Starting to model commercial
            "pos": Decimal("0.95"),         # ↓ Lower info uncertainty
            "short_interest": Decimal("1.00"),  # → Neutral
            "momentum": Decimal("1.00"),    # → Neutral
        },
        "regulatory": {  # PDUFA/approval phase
            "catalyst": Decimal("1.60"),    # ↑↑ Binary regulatory event
            "valuation": Decimal("1.25"),   # ↑ Commercial value crystallizes
            "financial": Decimal("1.15"),   # ↑ Launch funding critical
            "momentum": Decimal("1.10"),    # ↑ Pre-PDUFA positioning
            "clinical": Decimal("0.85"),    # ↓ Science mostly priced
            "pos": Decimal("0.90"),         # ↓ Low info uncertainty
            "short_interest": Decimal("0.95"),  # ↓ Less short interest usually
        },
        "commercial": {  # Post-approval, revenue generating
            "financial": Decimal("1.35"),   # ↑↑ Execution risk dominant
            "valuation": Decimal("1.30"),   # ↑↑ Fundamentals matter
            "momentum": Decimal("1.20"),    # ↑ Sales/trajectory drives price
            "catalyst": Decimal("0.85"),    # → Few binary events
            "clinical": Decimal("0.80"),    # ↓ Science priced in
            "pos": Decimal("0.85"),         # ↓ Low clinical uncertainty
            "short_interest": Decimal("0.90"),  # → Lower short interest
        },
        # Neutral fallback (no tilting)
        "none": {},
    },

    # Size tilts (multiplicative) - what matters at each market cap tier
    "size_tilts": {
        "micro": {  # <$300M
            "financial": Decimal("1.25"),   # ↑↑ Survivability is everything
            "catalyst": Decimal("1.15"),    # ↑ Binary events lifeline
            "short_interest": Decimal("1.15"),  # ↑ High short interest common
            "clinical": Decimal("1.05"),    # → Binary science risk
            "pos": Decimal("1.10"),         # → PoS more uncertain
            "valuation": Decimal("0.80"),   # ↓ No fundamentals
            "momentum": Decimal("0.85"),    # → Illiquid, noise-driven
        },
        "small": {  # $300M-$2B
            "financial": Decimal("1.10"),   # ↑ Capital discipline key
            "catalyst": Decimal("1.05"),    # → Events still material
            "short_interest": Decimal("1.05"),  # → Shorts active
            "valuation": Decimal("0.95"),   # → Starting to matter
            "momentum": Decimal("0.95"),    # → Some liquidity
            "clinical": Decimal("1.00"),    # → Neutral
            "pos": Decimal("1.00"),         # → Neutral
        },
        "mid": {  # $2B-$10B (neutral baseline)
            "clinical": Decimal("1.00"), "financial": Decimal("1.00"),
            "catalyst": Decimal("1.00"), "pos": Decimal("1.00"),
            "momentum": Decimal("1.00"), "valuation": Decimal("1.00"),
            "short_interest": Decimal("1.00"),
        },
        "large": {  # >$10B
            "valuation": Decimal("1.20"),   # ↑ Fundamental valuation
            "financial": Decimal("1.15"),   # ↑ Capital allocation key
            "momentum": Decimal("1.10"),    # ↑ Liquidity, institutional flows
            "catalyst": Decimal("0.85"),    # → Events less material
            "short_interest": Decimal("0.85"),  # ↓ Minimal short interest
            "clinical": Decimal("0.90"),    # → Diversified, priced in
            "pos": Decimal("0.95"),         # → Lower uncertainty
        },
        "unknown": {},  # No tilting if size unknown
    },

    # Clamping ranges (applied INDIVIDUALLY to stage and size multipliers)
    "clamp_min": Decimal("0.70"),
    "clamp_max": Decimal("1.40"),
    # Final combined multiplier clamp (safety net to prevent extreme combos)
    "combined_clamp_min": Decimal("0.55"),
    "combined_clamp_max": Decimal("1.60"),
}

# =============================================================================
# ENHANCEMENT 9: BAKER-STYLE FUNDAMENTAL-CONCENTRATED MODE
# =============================================================================
# Emulates Baker Bros' thesis-first, concentrated investment approach:
# 1. Core thesis (clinical+pos) is dominant and gating
# 2. Survivability (financial) is critical for execution
# 3. Valuation matters for risk/reward (not "cheapness")
# 4. Catalysts are timing, not thesis
# 5. Momentum/short_interest are overlays, not drivers

BAKER_STYLE_TILT_CONFIG = {
    "enabled": True,

    # Stage tilts - thesis dominates early, execution/valuation rise later
    "stage_tilts": {
        "early": {  # Preclinical → Phase 1: thesis is everything
            "clinical": Decimal("1.30"),    # ↑↑ Biology quality dominant
            "pos": Decimal("1.25"),         # ↑↑ Binary outcomes
            "financial": Decimal("1.25"),   # ↑ Survivability critical
            "valuation": Decimal("0.65"),   # ↓↓ No fundamentals yet
            "catalyst": Decimal("0.70"),    # ↓ Events far out, low quality
            "momentum": Decimal("0.50"),    # ↓↓ Noise, not signal
            "short_interest": Decimal("0.50"),  # ↓↓ Overlay only
        },
        "poc": {  # Phase 2 → Proof-of-concept: thesis still key, catalysts rise
            "clinical": Decimal("1.15"),    # ↑ PoC data critical
            "pos": Decimal("1.10"),         # ↑ Refined probability
            "financial": Decimal("1.00"),   # → Neutral
            "valuation": Decimal("0.85"),   # → Still speculative
            "catalyst": Decimal("1.20"),    # ↑ Readout events matter
            "momentum": Decimal("0.70"),    # ↓ Pre-readout noise
            "short_interest": Decimal("0.80"),  # ↓ Context only
        },
        "pivotal": {  # Phase 3 → Regulatory submission
            "clinical": Decimal("1.00"),    # → Mature dataset
            "pos": Decimal("0.95"),         # → Lower uncertainty
            "financial": Decimal("1.10"),   # ↑ Capital for launch
            "valuation": Decimal("1.10"),   # ↑ Starting to model commercial
            "catalyst": Decimal("1.00"),    # → Pivotal events
            "momentum": Decimal("0.80"),    # ↓ Context
            "short_interest": Decimal("0.90"),  # → Context
        },
        "regulatory": {  # PDUFA/approval phase: binary event
            "clinical": Decimal("0.85"),    # ↓ Science priced
            "pos": Decimal("0.85"),         # ↓ Binary now
            "financial": Decimal("1.15"),   # ↑ Launch funding
            "valuation": Decimal("1.25"),   # ↑ Commercial value crystallizes
            "catalyst": Decimal("1.40"),    # ↑↑ Binary regulatory event
            "momentum": Decimal("1.00"),    # → Pre-PDUFA positioning
            "short_interest": Decimal("0.95"),  # → Context
        },
        "commercial": {  # Post-approval: execution + fundamentals
            "clinical": Decimal("0.75"),    # ↓↓ Science priced in
            "pos": Decimal("0.70"),         # ↓↓ No clinical uncertainty
            "financial": Decimal("1.30"),   # ↑↑ Execution risk dominant
            "valuation": Decimal("1.40"),   # ↑↑ Fundamentals matter
            "catalyst": Decimal("0.60"),    # ↓↓ Few binary events
            "momentum": Decimal("1.20"),    # ↑ Sales trajectory drives price
            "short_interest": Decimal("0.80"),  # ↓ Context
        },
        "none": {},
    },

    # Size tilts - same structure as STAGE_SIZE_TILT_CONFIG
    "size_tilts": {
        "micro": {
            "financial": Decimal("1.30"),   # ↑↑ Survivability is everything
            "clinical": Decimal("1.10"),    # ↑ Binary science risk
            "pos": Decimal("1.10"),         # ↑ More uncertain
            "catalyst": Decimal("1.10"),    # ↑ Binary events lifeline
            "valuation": Decimal("0.70"),   # ↓↓ No fundamentals
            "momentum": Decimal("0.60"),    # ↓↓ Illiquid, noise
            "short_interest": Decimal("0.80"),  # ↓ Context
        },
        "small": {
            "financial": Decimal("1.15"),   # ↑ Capital discipline
            "clinical": Decimal("1.05"),    # → Neutral+
            "pos": Decimal("1.05"),         # → Neutral+
            "catalyst": Decimal("1.00"),    # → Neutral
            "valuation": Decimal("0.90"),   # → Starting to matter
            "momentum": Decimal("0.80"),    # ↓ Context
            "short_interest": Decimal("0.90"),  # → Context
        },
        "mid": {
            "clinical": Decimal("1.00"), "financial": Decimal("1.00"),
            "catalyst": Decimal("1.00"), "pos": Decimal("1.00"),
            "momentum": Decimal("1.00"), "valuation": Decimal("1.00"),
            "short_interest": Decimal("1.00"),
        },
        "large": {
            "valuation": Decimal("1.25"),   # ↑ Fundamental valuation
            "financial": Decimal("1.15"),   # ↑ Capital allocation
            "momentum": Decimal("1.15"),    # ↑ Institutional flows
            "clinical": Decimal("0.85"),    # ↓ Diversified, priced
            "pos": Decimal("0.90"),         # ↓ Lower uncertainty
            "catalyst": Decimal("0.80"),    # ↓ Events less material
            "short_interest": Decimal("0.85"),  # ↓ Minimal short interest
        },
        "unknown": {},
    },

    "clamp_min": Decimal("0.50"),
    "clamp_max": Decimal("1.50"),
    "combined_clamp_min": Decimal("0.40"),
    "combined_clamp_max": Decimal("1.80"),
}

# =============================================================================
# ENHANCEMENT 10: THESIS GATE (CLINICAL+POS ELIGIBILITY)
# =============================================================================
# "You can't rank top without a real thesis" - prevents momentum/value traps
# from floating to top ranks when clinical quality is weak.
#
# Implementation: Eligibility gate for top ranks
# - If (clinical + pos) / 2 < threshold → cap final score at ceiling
# - Stage-aware: stricter for early/poc, looser for commercial

THESIS_GATE_CONFIG = {
    "enabled": True,

    # Threshold: average of clinical + pos normalized scores
    # Below this = weak thesis, cap the final score
    "thresholds_by_stage": {
        "early": Decimal("58"),      # Strict: thesis must be strong
        "poc": Decimal("55"),        # Strict: thesis still dominant
        "pivotal": Decimal("50"),    # Moderate: execution matters too
        "regulatory": Decimal("48"), # Looser: binary event dominates
        "commercial": Decimal("45"), # Loosest: execution/valuation matter
        "late": Decimal("45"),       # Late-stage companies (Phase 3+/approved) - same as commercial
        "none": Decimal("50"),       # Default fallback
    },

    # Conviction override: bypass thesis gate if institutional signal is very strong
    # This allows high-conviction late-stage names to receive reinforcement
    "conviction_override": {
        "enabled": True,
        "min_tier1": 10,            # Require at least 10 tier-1 holders
        "min_conviction": Decimal("15"),  # Require conviction overlap >= 15
    },

    # Score ceiling when thesis gate triggers
    # This prevents weak-thesis names from ranking in top decile
    "ceiling_by_stage": {
        "early": Decimal("62"),      # Can't exceed ~60th percentile
        "poc": Decimal("65"),        # Can't exceed ~65th percentile
        "pivotal": Decimal("68"),    # Slightly more permissive
        "regulatory": Decimal("70"), # Even more permissive
        "commercial": Decimal("72"), # Most permissive
        "late": Decimal("72"),       # Late-stage - same as commercial
        "none": Decimal("65"),       # Default fallback
    },

    # Soft penalty (enabled) - applies smooth penalty instead of hard ceiling
    # Works regardless of score compression (ceiling approach fails with compressed scores)
    "use_soft_penalty": True,
    "soft_penalty_base": Decimal("0.92"),  # Multiply score by this at threshold
    "soft_penalty_floor": Decimal("0.80"), # Minimum multiplier (at 20pt gap below threshold)
}

# =============================================================================
# ENHANCEMENT 11: SMART MONEY REINFORCEMENT (OPTION D)
# =============================================================================
# Use Tier-1 overlap as an interaction coefficient that amplifies catalyst
# (and optionally momentum) contributions in PoC/Regulatory stages.
#
# Guardrails:
# - Only apply when smart-money confidence is high (≥0.65 or overlap ≥2 Tier-1s)
# - Only in PoC + Regulatory (event-alpha regimes)
# - Cap reinforcement tightly: ±5-10% max effect
# - Apply to component contributions, not final score (keeps attribution clean)

SMART_MONEY_REINFORCEMENT_CONFIG = {
    "enabled": True,  # Master switch (only active in BAKER_STYLE mode)

    # Stages where reinforcement applies (event-alpha regimes)
    "active_stages": ["poc", "regulatory"],

    # Minimum confidence/overlap to apply reinforcement
    "min_smart_money_confidence": Decimal("0.65"),
    "min_tier1_overlap": 2,

    # Reinforcement multipliers for component contributions
    # Positive overlap (Tier-1s holding) → amplify catalyst contribution
    "catalyst_reinforcement": {
        "strong_overlap_mult": Decimal("1.10"),   # 3+ Tier-1s → +10% catalyst contribution
        "moderate_overlap_mult": Decimal("1.05"), # 2 Tier-1s → +5% catalyst contribution
        "weak_overlap_mult": Decimal("1.00"),     # <2 Tier-1s → no change
        "negative_signal_mult": Decimal("0.93"),  # Selling signal → -7% catalyst contribution
    },

    # Momentum reinforcement (half the size of catalyst)
    "momentum_reinforcement": {
        "strong_overlap_mult": Decimal("1.05"),   # 3+ Tier-1s → +5% momentum contribution
        "moderate_overlap_mult": Decimal("1.02"), # 2 Tier-1s → +2% momentum contribution
        "weak_overlap_mult": Decimal("1.00"),     # <2 Tier-1s → no change
        "negative_signal_mult": Decimal("0.97"),  # Selling signal → -3% momentum contribution
    },

    # Tier-1 overlap thresholds
    "strong_overlap_threshold": 3,  # ≥3 Tier-1s = strong
    "moderate_overlap_threshold": 2, # ≥2 Tier-1s = moderate
}


# =============================================================================
# TYPES
# =============================================================================

class MonotonicCap:
    LIQUIDITY_FAIL_CAP = Decimal("35")
    LIQUIDITY_WARN_CAP = Decimal("60")
    RUNWAY_CRITICAL_CAP = Decimal("40")
    RUNWAY_WARNING_CAP = Decimal("55")
    DILUTION_SEVERE_CAP = Decimal("45")
    DILUTION_HIGH_CAP = Decimal("60")


class ScoringMode(str, Enum):
    """Scoring mode based on available data."""
    DEFAULT = "default"
    PARTIAL = "partial"
    ENHANCED = "enhanced"
    ADAPTIVE = "adaptive"
    BAKER_STYLE = "baker_style"  # Fundamental-concentrated (thesis-first)


class RunStatus(str, Enum):
    """Pipeline run status based on data coverage health."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAIL = "FAIL"


class NormalizationMethod(str, Enum):
    """Normalization method applied."""
    COHORT = "cohort"
    COHORT_WINSORIZED = "cohort_winsorized"
    COHORT_SHRINKAGE = "cohort_shrinkage"
    STAGE_FALLBACK = "stage_fallback"
    GLOBAL_FALLBACK = "global_fallback"
    NONE = "none"


@dataclass
class ComponentScore:
    """Individual component score with full breakdown."""
    name: str
    raw: Optional[Decimal]
    normalized: Optional[Decimal]
    confidence: Decimal
    weight_base: Decimal
    weight_effective: Decimal
    contribution: Decimal
    decay_factor: Optional[Decimal] = None
    regime_adjustment: Optional[Decimal] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Complete score breakdown for auditability."""
    version: str
    mode: str
    base_weights: Dict[str, str]
    regime_adjustments: Dict[str, str]
    effective_weights: Dict[str, str]
    components: List[Dict[str, Any]]
    enhancements: Dict[str, Any]
    penalties_and_gates: Dict[str, Any]
    interaction_terms: Dict[str, Any]
    final: Dict[str, str]
    normalization_method: str
    cohort_info: Dict[str, Any]
    hybrid_aggregation: Dict[str, str]


@dataclass
class V3ScoringResult:
    """Complete v3 scoring result for a single ticker."""
    ticker: str
    composite_score: Decimal
    composite_rank: int
    clinical_normalized: Decimal
    financial_normalized: Decimal
    catalyst_normalized: Decimal
    momentum_normalized: Optional[Decimal]
    valuation_normalized: Optional[Decimal]
    pos_normalized: Optional[Decimal]
    vol_adjustment: VolatilityAdjustment
    catalyst_decay: CatalystDecayResult
    interaction_terms: InteractionTerms
    smart_money_signal: SmartMoneySignal
    severity: Severity
    confidence_overall: Decimal
    uncertainty_penalty: Decimal
    caps_applied: List[Dict[str, Any]]
    effective_weights: Dict[str, Decimal]
    cohort_key: str
    normalization_method: NormalizationMethod
    flags: List[str]
    determinism_hash: str
    score_breakdown: ScoreBreakdown


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _coalesce(*vals: Any, default: Optional[Any] = None) -> Any:
    """Return the first value that is not None.

    Unlike `or` chains, this correctly handles falsy values like Decimal("0")
    which should be treated as valid scores, not missing data.

    Args:
        *vals: Variable number of values to check
        default: Value to return if all vals are None

    Returns:
        First non-None value, or default if all are None
    """
    for v in vals:
        if v is not None:
            return v
    return default


def _conviction_horizon_weight_boost(
    weight_base: Decimal,
    weight_effective: Decimal,
    conviction_overlap: Optional[float],
    tier1_count: int,
    days_to_catalyst: Optional[int],
    cohort_conviction_stats: Optional[Dict[str, float]],
) -> Tuple[Decimal, Dict[str, Any]]:
    """
    Compute an ADDITIVE boost to catalyst weight for high-conviction names.

    This overlay provides additional catalyst weight for high-conviction names
    with real catalysts in the 90-240 day window, on top of what the continuous
    confidence gate already provides.

    All math is done in Decimal to ensure determinism across platforms.

    Args:
        weight_base: Base catalyst weight before gating
        weight_effective: Current effective weight (after continuous gate)
        conviction_overlap: Conviction-weighted overlap score
        tier1_count: Number of tier-1 holders
        days_to_catalyst: Days to nearest catalyst event
        cohort_conviction_stats: Mean/std conviction for stage×size cohort

    Returns:
        Tuple of (boost amount, diagnostics dict)
    """
    ZERO = Decimal("0")
    diagnostics: Dict[str, Any] = {"skip_reason": None}

    # Only extend horizon for real catalysts in a bounded window [91, 240]
    if days_to_catalyst is None:
        diagnostics["skip_reason"] = "no_catalyst_date"
        return ZERO, diagnostics
    if days_to_catalyst <= 90:
        diagnostics["skip_reason"] = "catalyst_too_soon"
        return ZERO, diagnostics
    if days_to_catalyst > 240:
        diagnostics["skip_reason"] = "catalyst_too_far"
        return ZERO, diagnostics

    # Extract cohort stats as Decimal
    mean = _to_decimal((cohort_conviction_stats or {}).get("mean", 0)) or ZERO
    std = _to_decimal((cohort_conviction_stats or {}).get("std", 1)) or Decimal("1")
    if std < Decimal("0.000001"):
        std = Decimal("1")

    # Compute conviction z-score (all Decimal)
    conv = _to_decimal(conviction_overlap) or ZERO
    z = (conv - mean) / std if std > ZERO else ZERO

    # Require minimum conviction signal (tier1 >= 6 OR z-score >= 1.0)
    tier1 = tier1_count or 0
    if tier1 < 6 and z < Decimal("1"):
        diagnostics["skip_reason"] = "insufficient_conviction"
        diagnostics["tier1"] = tier1
        diagnostics["z_score"] = str(_quantize_score(z))
        return ZERO, diagnostics

    # tol: z≈0.5→0, z≈1.5→1 (linear ramp)
    tol = _clamp((z - Decimal("0.5")) / Decimal("1"), ZERO, Decimal("1"))

    # Tier1 count provides a floor for tolerance
    if tier1 >= 6:
        tol = max(tol, Decimal("0.75"))

    # Horizon factor: peaks around 150d, fades to 0 by 60d or 240d
    days_dec = Decimal(str(days_to_catalyst))
    horizon = Decimal("1") - abs(days_dec - Decimal("150")) / Decimal("90")
    horizon = _clamp(horizon, ZERO, Decimal("1"))

    # Additive boost: up to 25% of base weight for high-conviction names
    BOOST_FRAC = Decimal("0.25")
    boost = weight_base * BOOST_FRAC * tol * horizon

    # Cap total effective weight at 90% of base (don't over-boost)
    max_total = weight_base * Decimal("0.90")
    max_boost = max(ZERO, max_total - weight_effective)
    boost = min(boost, max_boost)

    # Quantize for determinism
    boost = _quantize_weight(boost)

    diagnostics["tol"] = str(_quantize_score(tol))
    diagnostics["horizon"] = str(_quantize_score(horizon))
    diagnostics["z_score"] = str(_quantize_score(z))
    diagnostics["tier1"] = tier1

    return boost, diagnostics


def _apply_conviction_horizon_overlay(
    *,
    weight_base: Decimal,
    weight_effective: Decimal,
    confidence_low: bool,
    thesis_gate_blocked: bool,
    conviction_overlap: Optional[float],
    tier1_count: int,
    days_to_catalyst: Optional[int],
    cohort_conviction_stats: Optional[Dict[str, float]],
) -> Tuple[Decimal, Dict[str, Any]]:
    """
    Apply conviction horizon overlay with all pre-conditions checked.

    This wrapper enforces:
    - Thesis gate is a HARD BLOCK (cannot bypass)
    - Confidence must be low (the root issue we're solving)
    - Delegates to _conviction_horizon_weight_boost for actual calculation

    Args:
        weight_base: Base catalyst weight before gating
        weight_effective: Current effective weight (after continuous gate)
        confidence_low: Whether catalyst confidence is below threshold
        thesis_gate_blocked: Whether thesis gate would block (hard block)
        conviction_overlap: Conviction-weighted overlap score
        tier1_count: Number of tier-1 holders
        days_to_catalyst: Days to nearest catalyst event
        cohort_conviction_stats: Mean/std conviction for stage×size cohort

    Returns:
        Tuple of (new_weight, diagnostics dict)
    """
    # Thesis gate is a HARD BLOCK - cannot be bypassed by conviction
    if thesis_gate_blocked:
        return weight_effective, {"applied": False, "skip_reason": "thesis_gate_blocked"}

    # Only apply when confidence is low (the root issue)
    if not confidence_low:
        return weight_effective, {"applied": False, "skip_reason": "confidence_not_low"}

    # Delegate to boost calculator
    boost, boost_diag = _conviction_horizon_weight_boost(
        weight_base=weight_base,
        weight_effective=weight_effective,
        conviction_overlap=conviction_overlap,
        tier1_count=tier1_count,
        days_to_catalyst=days_to_catalyst,
        cohort_conviction_stats=cohort_conviction_stats,
    )

    if boost <= Decimal("0"):
        skip_reason = boost_diag.get("skip_reason", "no_boost")
        return weight_effective, {"applied": False, "skip_reason": skip_reason, "boost_diagnostics": boost_diag}

    new_w = weight_effective + boost
    return new_w, {
        "applied": True,
        "weight_boost": str(boost),
        "weight_before": str(weight_effective),
        "weight_after": str(new_w),
        "days_to_catalyst": days_to_catalyst,
        "tier1_count": tier1_count,
        "boost_diagnostics": boost_diag,
    }


def _compute_catalyst_effective(
    catalyst_score_window: Optional[Decimal],
    catalyst_proximity_score: Optional[Decimal],
) -> Tuple[Decimal, bool, str]:
    """Compute effective catalyst score by blending window and proximity scores."""
    has_window = catalyst_score_window is not None
    has_proximity = catalyst_proximity_score is not None and catalyst_proximity_score > Decimal("0")

    if has_window and has_proximity:
        effective = (
            CATALYST_WINDOW_WEIGHT * catalyst_score_window +
            CATALYST_PROXIMITY_WEIGHT * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "full_blend"
    elif has_proximity and not has_window:
        effective = (
            CATALYST_DEFAULT_BASE * CATALYST_DEFAULT_SCORE +
            (Decimal("1") - CATALYST_DEFAULT_BASE) * catalyst_proximity_score
        )
        proximity_blended = True
        blend_mode = "proximity_only"
    elif has_window and not has_proximity:
        effective = catalyst_score_window
        proximity_blended = False
        blend_mode = "window_only"
    else:
        effective = CATALYST_DEFAULT_SCORE
        proximity_blended = False
        blend_mode = "default"

    effective = _clamp(effective, Decimal("0"), Decimal("100"))
    effective = _quantize_score(effective)

    return (effective, proximity_blended, blend_mode)


def _market_cap_bucket(market_cap_mm: Optional[Any]) -> str:
    """Classify market cap into bucket."""
    mcap = _to_decimal(market_cap_mm)
    if mcap is None:
        return "unknown"
    if mcap >= 10000:
        return "large"
    elif mcap >= 2000:
        return "mid"
    elif mcap >= 300:
        return "small"
    return "micro"


def _stage_bucket(lead_phase: Optional[str]) -> str:
    """Classify development stage into bucket."""
    if not lead_phase:
        return "early"
    phase = lead_phase.lower()
    if "3" in phase or "approved" in phase:
        return "late"
    elif "2" in phase:
        return "mid"
    return "early"


def _determine_stage_bucket_alpha(
    lead_phase: Optional[str],
    cat_event_type: Optional[str] = None,
    days_to_catalyst: Optional[int] = None,
) -> str:
    """
    Determine stage bucket with event-awareness for stage/size tilting.

    Priority: catalyst event type > lead phase > default 'pivotal'.

    Returns: 'early', 'poc', 'pivotal', 'regulatory', 'commercial', or 'none'
    """
    # Phase 1: Catalyst event type mapping (Module 3 event types + keywords)
    if cat_event_type:
        cat_type_upper = str(cat_event_type).upper()
        cat_type_lower = str(cat_event_type).lower()

        # Explicit M3 event type mapping (CT_* and FDA_* types)
        # FDA_PDUFA_DATE → regulatory (binary regulatory event)
        if cat_type_upper == "FDA_PDUFA_DATE":
            return "regulatory"
        # CT_RESULTS_POSTED → poc (actual data posted = repricing catalyst)
        if cat_type_upper == "CT_RESULTS_POSTED":
            return "poc"
        # CT_PRIMARY_COMPLETION → poc (readout imminent / analysis window)
        if cat_type_upper == "CT_PRIMARY_COMPLETION":
            return "poc"
        # CT_STUDY_COMPLETION → pivotal (later-stage completion more common)
        if cat_type_upper == "CT_STUDY_COMPLETION":
            return "pivotal"

        # Regulatory events (keyword matching)
        if any(kw in cat_type_lower for kw in [
            "pdufa", "adcom", "nda", "bla", "fda_pdufa", "fda_adcom",
            "filing", "approval", "regulatory", "label", "crl",
        ]):
            return "regulatory"

        # Pivotal/Phase 3 events
        if any(kw in cat_type_lower for kw in [
            "phase 3", "phase_3", "ph3", "pivotal", "registrational",
            "phase iii", "p3", "primary_completion",
        ]):
            return "pivotal"

        # Proof-of-concept/Phase 2 events
        if any(kw in cat_type_lower for kw in [
            "phase 2", "phase_2", "ph2", "interim", "readout", "data",
            "phase ii", "p2", "topline", "results",
        ]):
            return "poc"

        # Early stage events
        if any(kw in cat_type_lower for kw in [
            "phase 1", "phase_1", "ph1", "preclinical", "ind",
            "phase i", "p1", "first_patient", "dose_escalation",
        ]):
            return "early"

        # Commercial events
        if any(kw in cat_type_lower for kw in [
            "launch", "sales", "earnings", "guidance", "partnership",
            "acquisition", "commercial", "revenue",
        ]):
            return "commercial"

    # Phase 2: Lead phase fallback
    if lead_phase:
        phase_lower = lead_phase.lower()
        if any(kw in phase_lower for kw in ["preclinical", "pre-clinical"]):
            return "early"
        elif "phase 1" in phase_lower or "phase_1" in phase_lower or "phase i" in phase_lower:
            return "early"
        elif "phase 2" in phase_lower or "phase_2" in phase_lower or "phase ii" in phase_lower:
            return "poc"
        elif "phase 3" in phase_lower or "phase_3" in phase_lower or "phase iii" in phase_lower:
            return "pivotal"
        elif any(kw in phase_lower for kw in ["approved", "commercial", "launched", "marketed"]):
            return "commercial"
        elif any(kw in phase_lower for kw in ["nda", "bla", "filing", "regulatory", "pdufa"]):
            return "regulatory"

    # Phase 3: Days to catalyst heuristic (if no event type but we have timing)
    if days_to_catalyst is not None:
        try:
            days = int(days_to_catalyst)
            if days < 60:
                return "regulatory"  # Imminent event likely regulatory
            elif days < 120:
                return "pivotal"     # Near-term likely pivotal
            elif days < 270:
                return "poc"         # Medium-term likely PoC
        except (ValueError, TypeError):
            pass

    # Default: 'pivotal' (most neutral, midpoint of development)
    return "pivotal"


def _apply_stage_size_tilts(
    base_weights: Dict[str, Decimal],
    stage_bucket: str,
    size_bucket: str,
    enable_tilting: bool = True,
    tilt_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Decimal], Dict[str, Any]]:
    """
    Apply stage × size tilts to base weights.

    Tilts are multiplicative and individually clamped to prevent extreme values.
    Weights are renormalized after tilting to sum to 1.0.

    Args:
        base_weights: Component weights before tilting
        stage_bucket: Development stage ('early', 'poc', 'pivotal', 'regulatory', 'commercial')
        size_bucket: Market cap bucket ('micro', 'small', 'mid', 'large')
        enable_tilting: Master switch to disable tilting
        tilt_config: Optional tilt configuration (defaults to STAGE_SIZE_TILT_CONFIG)

    Returns:
        Tuple of (tilted_weights, diagnostics_dict)
    """
    config = tilt_config if tilt_config is not None else STAGE_SIZE_TILT_CONFIG

    # Return unchanged weights if tilting disabled
    if not enable_tilting or not config.get("enabled", True) or stage_bucket == "none":
        return base_weights.copy(), {
            "stage": stage_bucket,
            "size": size_bucket,
            "tilt_applied": False,
            "stage_tilts": {},
            "size_tilts": {},
            "combined_multipliers": {},
            "clamp_hits": {"stage": 0, "size": 0, "combined": 0},
        }

    stage_tilt = config["stage_tilts"].get(stage_bucket, {})
    size_tilt = config["size_tilts"].get(size_bucket, {})

    clamp_min = config["clamp_min"]
    clamp_max = config["clamp_max"]
    combined_min = config["combined_clamp_min"]
    combined_max = config["combined_clamp_max"]

    tilted = {}
    diagnostics = {
        "stage": stage_bucket,
        "size": size_bucket,
        "tilt_applied": False,
        "stage_tilts": {},
        "size_tilts": {},
        "combined_multipliers": {},
        "clamp_hits": {"stage": 0, "size": 0, "combined": 0},
    }

    # Apply tilts with individual clamping
    for component, base_w in base_weights.items():
        # Get stage multiplier (default 1.0)
        stage_mult_raw = Decimal(str(stage_tilt.get(component, "1.0")))
        stage_mult = _clamp(stage_mult_raw, clamp_min, clamp_max)
        if stage_mult != stage_mult_raw:
            diagnostics["clamp_hits"]["stage"] += 1

        # Get size multiplier (default 1.0)
        size_mult_raw = Decimal(str(size_tilt.get(component, "1.0")))
        size_mult = _clamp(size_mult_raw, clamp_min, clamp_max)
        if size_mult != size_mult_raw:
            diagnostics["clamp_hits"]["size"] += 1

        # Combined multiplier with safety clamp
        combined_mult_raw = stage_mult * size_mult
        combined_mult = _clamp(combined_mult_raw, combined_min, combined_max)
        if combined_mult != combined_mult_raw:
            diagnostics["clamp_hits"]["combined"] += 1

        # Apply tilt
        tilted[component] = base_w * combined_mult

        # Store diagnostics for non-neutral tilts
        if stage_mult != Decimal("1.0"):
            diagnostics["stage_tilts"][component] = str(stage_mult)
        if size_mult != Decimal("1.0"):
            diagnostics["size_tilts"][component] = str(size_mult)
        if combined_mult != Decimal("1.0"):
            diagnostics["combined_multipliers"][component] = str(combined_mult)

    # Renormalize to sum = 1.0 (preserve high precision)
    total = sum(tilted.values())
    if total > EPS:
        tilted = {k: v / total for k, v in tilted.items()}

    # Check if any tilts were applied
    diagnostics["tilt_applied"] = bool(diagnostics["combined_multipliers"])

    return tilted, diagnostics


# =============================================================================
# SMART MONEY ENHANCEMENT FUNCTIONS (A+B+C+E)
# =============================================================================

def compute_smart_money_cohort_stats(
    combined_records: List[Dict],
) -> Dict[Tuple[str, str], Tuple[Decimal, Decimal]]:
    """
    Compute mean and std of overlap counts by (stage_bucket, size_bucket).
    Used for cohort normalization (Option C).

    Returns:
        Dict mapping (stage, size) tuple to (mean, std) of overlap counts.
    """
    config = SMART_MONEY_ENHANCEMENT_CONFIG
    min_cohort_size = config["cohort_normalization"]["min_cohort_size"]

    cohort_data: Dict[Tuple[str, str], List[Decimal]] = {}

    # Group by cohort
    for rec in combined_records:
        # Get stage bucket from enhanced alpha detection
        stage = rec.get("stage_bucket_alpha", rec.get("stage_bucket", "pivotal"))
        size = rec.get("market_cap_bucket", "mid")
        coinvest = rec.get("coinvest", {})
        overlap = coinvest.get("coinvest_overlap_count", 0)

        key = (stage, size)
        cohort_data.setdefault(key, []).append(Decimal(str(overlap)))

    # Compute stats for cohorts with enough data
    cohort_stats: Dict[Tuple[str, str], Tuple[Decimal, Decimal]] = {}
    for key, overlaps in cohort_data.items():
        if len(overlaps) >= min_cohort_size:
            mean = sum(overlaps) / len(overlaps)
            # Population std
            variance = sum((x - mean) ** 2 for x in overlaps) / len(overlaps)
            # sqrt for Decimal
            if variance > 0:
                std = variance.sqrt()
            else:
                std = Decimal("1.0")
            cohort_stats[key] = (mean, std)

    return cohort_stats


def compute_enhanced_smart_money_signal_v2(
    coinvest_data: Dict,
    stage_bucket: str,
    size_bucket: str,
    cohort_stats: Optional[Dict[Tuple[str, str], Tuple[Decimal, Decimal]]] = None,
) -> Dict[str, Any]:
    """
    Enhanced smart money signal with:
    A) Stage×size dependent base weight
    B) Expanded score range (20-85)
    C) Cohort-normalized overlap (z-score within stage×size)
    E) Confidence from overlap count

    Returns enhanced signal dict with score, weight, confidence, and diagnostics.
    """
    config = SMART_MONEY_ENHANCEMENT_CONFIG

    # Extract base data
    overlap_count = coinvest_data.get("coinvest_overlap_count", 0)
    holders = coinvest_data.get("coinvest_holders", [])
    position_changes = coinvest_data.get("position_changes", {})
    holder_tiers = coinvest_data.get("holder_tiers", {})
    tier1_count_raw = coinvest_data.get("tier1_count", 0)

    # Get Tier-1 holders from holder_tiers dict
    tier1_holders = [h for h, tier in holder_tiers.items() if tier == 1]
    tier1_count = len(tier1_holders) if tier1_holders else tier1_count_raw

    # Get position change direction
    increasing_holders = [h for h, change in position_changes.items()
                         if change in ("INCREASE", "NEW")]
    decreasing_holders = [h for h, change in position_changes.items()
                         if change in ("DECREASE", "EXIT")]

    # =========================================================================
    # C: COHORT-NORMALIZED OVERLAP (prevents size bias)
    # =========================================================================
    cohort_key = (stage_bucket, size_bucket)
    z_score = Decimal("0")

    if (config["cohort_normalization"]["enabled"] and
        cohort_stats and cohort_key in cohort_stats):
        cohort_mean, cohort_std = cohort_stats[cohort_key]
        if cohort_std > Decimal("0.1"):  # Avoid division by tiny std
            z_score = (Decimal(str(overlap_count)) - cohort_mean) / cohort_std
            # Clamp z-score to [-2, 2] to avoid extreme values
            z_score = _clamp(z_score, Decimal("-2.0"), Decimal("2.0"))

    # Convert z-score to base score component (centered at neutral)
    cohort_component = (config["score_range"]["neutral"] +
                        z_score * config["cohort_normalization"]["zscore_to_score_factor"])

    # =========================================================================
    # B: EXPANDED SCORE RANGE WITH HOLDER-BASED BOOSTS
    # =========================================================================
    base_score = cohort_component

    # Tier-1 holder boost
    tier1_boost = min(
        Decimal(str(tier1_count)) * config["score_range"]["per_tier1_holder_boost"],
        config["score_range"]["max_boost_from_holders"]
    )

    # Position increase boost
    increase_boost = (Decimal(str(len(increasing_holders))) *
                      config["score_range"]["per_position_increase_boost"])

    # Position decrease penalty
    decrease_penalty = (Decimal(str(len(decreasing_holders))) *
                        config["score_range"]["per_position_increase_boost"] * Decimal("-1"))

    # Calculate raw boosted score
    raw_boosted = base_score + tier1_boost + increase_boost + decrease_penalty

    # Apply expanded range
    final_score = _clamp(
        raw_boosted,
        config["score_range"]["min"],
        config["score_range"]["max"]
    )

    # =========================================================================
    # E: CONFIDENCE FROM OVERLAP
    # =========================================================================
    conf_config = config["confidence_from_overlap"]
    base_conf = conf_config["base_confidence"]

    overlap_boost = min(
        Decimal(str(overlap_count)) * conf_config["per_overlap_boost"],
        Decimal("0.3")
    )
    tier1_confidence_boost = min(
        Decimal(str(tier1_count)) * conf_config["per_tier1_boost"],
        Decimal("0.4")
    )

    final_confidence = base_conf + overlap_boost + tier1_confidence_boost
    final_confidence = _clamp(
        final_confidence,
        Decimal("0.3"),
        conf_config["max_confidence"]
    )

    # =========================================================================
    # A: STAGE×SIZE DYNAMIC WEIGHT
    # =========================================================================
    stage_weights = config["base_weights_by_stage_size"].get(stage_bucket, {})
    base_weight = stage_weights.get(size_bucket, Decimal("0.05"))

    # Adjust weight based on signal strength (more Tier-1 → higher weight)
    weight_multiplier = Decimal("1.0") + (Decimal(str(tier1_count)) * Decimal("0.1"))
    dynamic_weight = base_weight * weight_multiplier

    # Cap at 15% max weight
    dynamic_weight = _clamp(dynamic_weight, Decimal("0"), Decimal("0.15"))

    # Build flags
    flags = []
    if tier1_count > 0:
        flags.append(f"tier1_{tier1_count}")
    if increasing_holders:
        flags.append(f"sm_increasing_{len(increasing_holders)}")
    if decreasing_holders:
        flags.append(f"sm_decreasing_{len(decreasing_holders)}")
    if z_score != Decimal("0"):
        flags.append(f"sm_zscore_{float(z_score):.2f}")

    return {
        "score": final_score,
        "weight": dynamic_weight,
        "confidence": final_confidence,
        "tier1_count": tier1_count,
        "tier1_holders": tier1_holders,
        "increasing_holders": increasing_holders,
        "decreasing_holders": decreasing_holders,
        "overlap_count": overlap_count,
        "cohort_z_score": str(z_score) if z_score != Decimal("0") else None,
        "cohort_key": f"{stage_bucket}_{size_bucket}",
        "base_score": str(cohort_component),
        "tier1_boost": str(tier1_boost),
        "increase_boost": str(increase_boost),
        "decrease_penalty": str(decrease_penalty),
        "final_score": str(final_score),
        "flags": flags,
    }


def _quarter_from_date(d: date) -> str:
    """Convert date to quarter string."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _get_worst_severity(severities: List[str]) -> Severity:
    """Get worst severity from list."""
    priority = {Severity.SEV3: 3, Severity.SEV2: 2, Severity.SEV1: 1, Severity.NONE: 0}
    worst = Severity.NONE
    for s in severities:
        try:
            sev = Severity(s)
            if priority[sev] > priority[worst]:
                worst = sev
        except (ValueError, KeyError):
            continue
    return worst


def _rank_normalize_winsorized(values: List[Decimal]) -> Tuple[List[Decimal], bool]:
    """Winsorized percentile rank normalization."""
    n = len(values)
    if n == 0:
        return [], False
    if n == 1:
        return [Decimal("50")], False

    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])

    ranks = [Decimal("0")] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = Decimal(str((i + j - 1) / 2))
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j

    raw_percentiles = []
    for r in ranks:
        pct = (r / Decimal(str(n - 1))) * Decimal("100") if n > 1 else Decimal("50")
        raw_percentiles.append(pct)

    winsorization_applied = False
    result = []
    for pct in raw_percentiles:
        if pct < WINSOR_LOW or pct > WINSOR_HIGH:
            winsorization_applied = True
        clipped = _clamp(pct, WINSOR_LOW, WINSOR_HIGH)
        # Map [P5, P95] to [5, 95] instead of [0, 100] to eliminate zero cliff
        rescaled = Decimal("5") + ((clipped - WINSOR_LOW) / (WINSOR_HIGH - WINSOR_LOW)) * Decimal("90")
        result.append(_quantize_score(rescaled))

    return result, winsorization_applied


def _extract_confidence_financial(fin_data: Dict) -> Decimal:
    """Extract confidence for financial score."""
    conf = _to_decimal(fin_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    state = fin_data.get("financial_data_state", "NONE")
    return {"FULL": Decimal("1.0"), "PARTIAL": Decimal("0.7"),
            "MINIMAL": Decimal("0.4"), "NONE": Decimal("0.1")}.get(state, Decimal("0.5"))


def _extract_confidence_clinical(clin_data: Dict) -> Decimal:
    """Extract confidence for clinical score."""
    conf = _to_decimal(clin_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    trial_count = clin_data.get("trial_count", 0)
    has_lead = clin_data.get("lead_phase") is not None
    has_score = clin_data.get("clinical_score") is not None
    base = Decimal("0.3")
    if has_score:
        base = Decimal("0.5")
    if trial_count > 0:
        base += Decimal("0.2")
    if trial_count > 3:
        base += Decimal("0.1")
    if has_lead:
        base += Decimal("0.2")
    return _clamp(base, Decimal("0"), Decimal("1"))


def _extract_confidence_catalyst(cat_data: Any) -> Decimal:
    """Extract confidence for catalyst score."""
    if not cat_data:
        return Decimal("0.3")

    CONFIDENCE_MAP = {
        "HIGH": Decimal("0.70"),
        "MED": Decimal("0.50"),
        "LOW": Decimal("0.35"),
        "UNKNOWN": Decimal("0.30"),
    }

    if hasattr(cat_data, 'catalyst_confidence') and hasattr(cat_data.catalyst_confidence, 'value'):
        conf_str = cat_data.catalyst_confidence.value
        if conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

    if isinstance(cat_data, dict):
        # Check for direct override confidence first (from catalyst_overrides.json)
        scores = cat_data.get("scores", {})
        override_conf = scores.get("override_confidence")
        if override_conf:
            return _clamp(_to_decimal(override_conf), Decimal("0"), Decimal("1"))

        integration = cat_data.get("integration", {})
        conf_str = integration.get("catalyst_confidence")
        if conf_str and conf_str in CONFIDENCE_MAP:
            return CONFIDENCE_MAP[conf_str]

        event_summary = cat_data.get("event_summary", {})
        events_total = event_summary.get("events_total", 0)
        if events_total > 0 and conf_str is None:
            return Decimal("0.50")

    if hasattr(cat_data, 'n_high_confidence'):
        n_high = cat_data.n_high_confidence
        n_events = cat_data.events_detected_total
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    if isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        n_high = scores.get("n_high_confidence", 0)
        n_events = scores.get("events_detected_total", 0)
        if n_events > 0:
            high_ratio = Decimal(str(n_high)) / Decimal(str(max(n_events, 1)))
            return _clamp(Decimal("0.4") + high_ratio * Decimal("0.5"), Decimal("0"), Decimal("1"))

    return Decimal("0.3")


def _extract_confidence_pos(pos_data: Optional[Dict]) -> Decimal:
    """Extract confidence for PoS score.

    Reads pos_confidence from PoS engine (stage + indication adjusted).
    Falls back to 0.7 if pos_score exists but confidence missing.
    """
    if not pos_data:
        return Decimal("0")
    # Primary: read pos_confidence from PoS engine
    conf = _to_decimal(pos_data.get("pos_confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    # Fallback: legacy "confidence" field
    conf = _to_decimal(pos_data.get("confidence"))
    if conf is not None:
        return _clamp(conf, Decimal("0"), Decimal("1"))
    return Decimal("0.7") if pos_data.get("pos_score") is not None else Decimal("0")


def _extract_confidence_short_interest(si_data: Optional[Dict]) -> Decimal:
    """Extract confidence for short interest signal.

    Confidence is based on:
    - Data availability (base 0.5 if we have any SI data)
    - Signal strength (higher SI% = higher confidence in the signal)
    - Data freshness (if status is SUCCESS)
    """
    if not si_data:
        return Decimal("0")

    if si_data.get("status") == "INSUFFICIENT_DATA":
        return Decimal("0.1")

    # Base confidence if we have data
    conf = Decimal("0.5")

    # Boost for high squeeze potential (strong signal)
    squeeze = si_data.get("squeeze_potential", "LOW")
    if squeeze == "EXTREME":
        conf += Decimal("0.3")
    elif squeeze == "HIGH":
        conf += Decimal("0.2")
    elif squeeze == "MODERATE":
        conf += Decimal("0.1")

    # Boost if we have trend data
    if si_data.get("trend_direction") not in (None, "UNKNOWN"):
        conf += Decimal("0.1")

    return _clamp(conf, Decimal("0"), Decimal("1"))


def _apply_monotonic_caps(
    score: Decimal,
    liquidity_gate_status: Optional[str],
    runway_months: Optional[Decimal],
    dilution_risk_bucket: Optional[str],
) -> Tuple[Decimal, List[Dict[str, Any]]]:
    """Apply monotonic caps based on risk gates."""
    caps_applied = []
    capped = score

    if liquidity_gate_status == "FAIL":
        if score > MonotonicCap.LIQUIDITY_FAIL_CAP:
            caps_applied.append({"reason": "liquidity_gate_fail", "cap": str(MonotonicCap.LIQUIDITY_FAIL_CAP)})
            capped = min(capped, MonotonicCap.LIQUIDITY_FAIL_CAP)
    elif liquidity_gate_status == "WARN":
        if score > MonotonicCap.LIQUIDITY_WARN_CAP:
            caps_applied.append({"reason": "liquidity_gate_warn", "cap": str(MonotonicCap.LIQUIDITY_WARN_CAP)})
            capped = min(capped, MonotonicCap.LIQUIDITY_WARN_CAP)

    if runway_months is not None:
        if runway_months < Decimal("6") and capped > MonotonicCap.RUNWAY_CRITICAL_CAP:
            caps_applied.append({"reason": "runway_critical", "cap": str(MonotonicCap.RUNWAY_CRITICAL_CAP)})
            capped = min(capped, MonotonicCap.RUNWAY_CRITICAL_CAP)
        elif runway_months < Decimal("12") and capped > MonotonicCap.RUNWAY_WARNING_CAP:
            caps_applied.append({"reason": "runway_warning", "cap": str(MonotonicCap.RUNWAY_WARNING_CAP)})
            capped = min(capped, MonotonicCap.RUNWAY_WARNING_CAP)

    if dilution_risk_bucket == "SEVERE" and capped > MonotonicCap.DILUTION_SEVERE_CAP:
        caps_applied.append({"reason": "dilution_severe", "cap": str(MonotonicCap.DILUTION_SEVERE_CAP)})
        capped = min(capped, MonotonicCap.DILUTION_SEVERE_CAP)
    elif dilution_risk_bucket == "HIGH" and capped > MonotonicCap.DILUTION_HIGH_CAP:
        caps_applied.append({"reason": "dilution_high", "cap": str(MonotonicCap.DILUTION_HIGH_CAP)})
        capped = min(capped, MonotonicCap.DILUTION_HIGH_CAP)

    return _quantize_score(capped), caps_applied


def _compute_determinism_hash(
    ticker: str,
    version: str,
    mode: str,
    base_weights: Dict[str, Decimal],
    effective_weights: Dict[str, Decimal],
    component_scores: List[ComponentScore],
    enhancements: Dict[str, Any],
    final_score: Decimal,
) -> str:
    """Compute deterministic hash for audit trail."""
    payload = {
        "ticker": ticker,
        "version": version,
        "mode": mode,
        "base_weights": {k: str(v) for k, v in sorted(base_weights.items())},
        "effective_weights": {k: str(v) for k, v in sorted(effective_weights.items())},
        "components": sorted([
            {"name": c.name, "raw": str(c.raw) if c.raw is not None else None,
             "normalized": str(c.normalized) if c.normalized is not None else None,
             "contribution": str(c.contribution)}
            for c in component_scores
        ], key=lambda x: x["name"]),
        "enhancements": {k: str(v) if isinstance(v, Decimal) else v for k, v in sorted(enhancements.items())},
        "final_score": str(final_score),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _enrich_with_coinvest(ticker: str, coinvest_signals: dict, as_of_date: date) -> dict:
    """Look up co-invest signal and return overlay fields with PIT safety."""
    signal = coinvest_signals.get(ticker)
    if not signal:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
            # Conviction fields (Baker-style) - defaults
            "conviction_overlap": None,
            "tier1_conviction_overlap": None,
            "tier1_count": 0,
            "max_tier1_position_pct": None,
            "days_since_latest_filing": None,
        }

    if isinstance(signal, dict) and "coinvest_overlap_count" in signal:
        raw_tiers = signal.get("holder_tiers", {})
        normalized_tiers = {}
        for holder_id, tier_info in raw_tiers.items():
            if isinstance(tier_info, dict):
                normalized_tiers[holder_id] = tier_info.get("tier", 2)
            elif isinstance(tier_info, int):
                normalized_tiers[holder_id] = tier_info
            else:
                normalized_tiers[holder_id] = 2

        return {
            "coinvest_overlap_count": signal.get("coinvest_overlap_count", 0),
            "coinvest_holders": signal.get("coinvest_holders", []),
            "coinvest_usable": signal.get("coinvest_overlap_count", 0) > 0,
            "position_changes": signal.get("position_changes", {}),
            "holder_tiers": normalized_tiers,
            # NEW: Conviction fields (Baker-style)
            "conviction_overlap": signal.get("conviction_overlap"),
            "tier1_conviction_overlap": signal.get("tier1_conviction_overlap"),
            "tier1_count": signal.get("tier1_count", 0),
            "max_tier1_position_pct": signal.get("max_tier1_position_pct"),
            "days_since_latest_filing": signal.get("days_since_latest_filing"),
        }

    if not hasattr(signal, 'positions'):
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
            "conviction_overlap": None,
            "tier1_conviction_overlap": None,
            "tier1_count": 0,
            "max_tier1_position_pct": None,
            "days_since_latest_filing": None,
        }

    pit_positions = [p for p in signal.positions if p.filing_date < as_of_date]
    if not pit_positions:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_usable": False,
            "position_changes": {},
            "holder_tiers": {},
            "conviction_overlap": None,
            "tier1_conviction_overlap": None,
            "tier1_count": 0,
            "max_tier1_position_pct": None,
            "days_since_latest_filing": None,
        }

    holders = sorted(set(p.manager_name for p in pit_positions))

    holder_tiers: Dict[str, int] = {}
    for p in pit_positions:
        manager_name = p.manager_name
        if manager_name not in holder_tiers:
            tier = getattr(p, 'manager_tier', None)
            if tier is not None:
                holder_tiers[manager_name] = tier

    position_changes = {}
    for p in pit_positions:
        change_type = getattr(p, 'change_type', None)
        if change_type is not None:
            position_changes[p.manager_name] = change_type

    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": _quarter_from_date(pit_positions[0].report_date),
        "coinvest_usable": True,
        "position_changes": position_changes,
        "holder_tiers": holder_tiers,
        # Conviction fields (Baker-style) - not available in position-based format
        "conviction_overlap": None,
        "tier1_conviction_overlap": None,
        "tier1_count": sum(1 for t in holder_tiers.values() if t == 1),
        "max_tier1_position_pct": None,
        "days_since_latest_filing": None,
    }


def _apply_cohort_normalization_v3(
    members: List[Dict],
    global_stats: Dict[str, Tuple[Decimal, Decimal]],
    include_pos: bool = False,
    use_shrinkage: bool = True,
) -> NormalizationMethod:
    """Apply V3 cohort normalization with optional shrinkage."""
    if not members:
        return NormalizationMethod.NONE

    n = len(members)

    clin_scores = [m["clinical_raw"] or Decimal("0") for m in members]
    fin_scores = [m["financial_raw"] or Decimal("0") for m in members]
    cat_scores = [m["catalyst_raw"] or Decimal("0") for m in members]

    if n >= MIN_COHORT_SIZE:
        if use_shrinkage and n < 20:
            clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
            fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
            cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

            clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
            fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
            cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

            method = NormalizationMethod.COHORT_SHRINKAGE
        else:
            clin_norm, clin_w = _rank_normalize_winsorized(clin_scores)
            fin_norm, fin_w = _rank_normalize_winsorized(fin_scores)
            cat_norm, cat_w = _rank_normalize_winsorized(cat_scores)

            method = NormalizationMethod.COHORT_WINSORIZED if (clin_w or fin_w or cat_w) else NormalizationMethod.COHORT
    else:
        clin_mean, clin_std = global_stats.get("clinical", (Decimal("50"), Decimal("20")))
        fin_mean, fin_std = global_stats.get("financial", (Decimal("50"), Decimal("20")))
        cat_mean, cat_std = global_stats.get("catalyst", (Decimal("50"), Decimal("20")))

        clin_norm, _ = shrinkage_normalize(clin_scores, clin_mean, clin_std)
        fin_norm, _ = shrinkage_normalize(fin_scores, fin_mean, fin_std)
        cat_norm, _ = shrinkage_normalize(cat_scores, cat_mean, cat_std)

        method = NormalizationMethod.GLOBAL_FALLBACK

    pos_norm = None
    if include_pos:
        pos_scores = [m.get("pos_raw") or Decimal("0") for m in members]
        if any(p > 0 for p in pos_scores):
            if n >= MIN_COHORT_SIZE:
                pos_norm, _ = _rank_normalize_winsorized(pos_scores)
            else:
                pos_mean, pos_std = global_stats.get("pos", (Decimal("50"), Decimal("20")))
                pos_norm, _ = shrinkage_normalize(pos_scores, pos_mean, pos_std)

    for i, m in enumerate(members):
        m["clinical_normalized"] = clin_norm[i]
        m["financial_normalized"] = fin_norm[i]
        m["catalyst_normalized"] = cat_norm[i]
        m["pos_normalized"] = pos_norm[i] if pos_norm else (m.get("pos_raw") or Decimal("0"))
        m["normalization_method"] = method

    return method


def _compute_global_stats(combined: List[Dict]) -> Dict[str, Tuple[Decimal, Decimal]]:
    """Compute global mean and std for each component."""
    stats = {}

    for component in ["clinical", "financial", "catalyst", "pos"]:
        key = f"{component}_raw"
        values = [_to_decimal(m.get(key)) for m in combined if m.get(key) is not None]

        if values:
            mean = sum(values) / Decimal(len(values))
            variance = sum((v - mean) ** 2 for v in values) / Decimal(len(values))
            std = variance.sqrt() if variance > 0 else Decimal("1")
            stats[component] = (mean, std)
        else:
            stats[component] = (Decimal("50"), Decimal("20"))

    return stats


# =============================================================================
# ENHANCEMENT 1: HARD REGIME GATING
# =============================================================================

def apply_regime_gates(
    normalized_scores: Dict[str, Decimal],
    regime: str,
) -> Tuple[Dict[str, Decimal], Dict[str, Any], List[str]]:
    """
    Apply hard regime-based gating to normalized scores.

    In BEAR regime: Cap momentum upside, cap valuation, amplify financial penalties
    In BULL regime: Allow full momentum, boost catalysts, soften penalties

    Args:
        normalized_scores: Dict of component name -> normalized score (0-100)
        regime: Current market regime ("BULL", "BEAR", "NEUTRAL")

    Returns:
        Tuple of (gated_scores, config_applied, flags)
    """
    gated = normalized_scores.copy()
    config = REGIME_GATE_CONFIG.get(regime.upper(), {})
    flags = []

    if not config:
        return gated, {}, flags

    # Gate momentum in BEAR regime
    if "momentum_cap_pct" in config and "momentum" in gated and gated["momentum"] is not None:
        cap_pct = config["momentum_cap_pct"]
        if cap_pct < Decimal("1.0"):
            original = gated["momentum"]
            # Cap deviation from neutral: max_deviation = (score - 50) * cap_pct
            deviation = original - Decimal("50")
            if deviation > 0:  # Only cap upside momentum
                max_deviation = deviation * cap_pct
                max_momentum = Decimal("50") + max_deviation
                if original > max_momentum:
                    gated["momentum"] = _quantize_score(max_momentum)
                    flags.append("regime_momentum_gated")

    # Cap valuation upside in BEAR regime
    if "valuation_upside_cap" in config and "valuation" in gated and gated["valuation"] is not None:
        cap = config["valuation_upside_cap"]
        if gated["valuation"] > cap:
            gated["valuation"] = cap
            flags.append("regime_valuation_capped")

    # Track financial penalty multiplier for downstream use
    if "financial_penalty_mult" in config:
        flags.append("regime_financial_amplified" if config["financial_penalty_mult"] > Decimal("1.0") else "regime_financial_softened")

    return gated, config, flags


# =============================================================================
# ENHANCEMENT 2: EXISTENTIAL FLAW DETECTION
# =============================================================================

def detect_existential_flaws(
    fin_data: Dict,
    clin_data: Dict,
    cat_data: Any,
) -> List[str]:
    """
    Detect existential flaws that warrant hard score caps.

    Existential flaws are binary risk factors that, if present, should prevent
    a security from scoring too high regardless of other positive factors.

    Args:
        fin_data: Financial data dict
        clin_data: Clinical data dict
        cat_data: Catalyst data (object or dict)

    Returns:
        List of existential flaw identifiers
    """
    flaws = []
    config = EXISTENTIAL_FLAW_CONFIG

    # Flaw 1: Critical runway (< 9 months)
    runway = _to_decimal(fin_data.get("runway_months"))
    if runway is not None and runway < config["runway_critical_months"]:
        flaws.append("runway_existential")

    # Flaw 2: Binary clinical risk (single asset in early phase)
    trial_count = clin_data.get("trial_count", 0)
    lead_phase = clin_data.get("lead_phase", "").lower() if clin_data.get("lead_phase") else ""

    # Check if in risky early phases
    is_early_phase = any(phase in lead_phase for phase in config["binary_clinical_risk_phases"])

    if trial_count <= 1 and is_early_phase:
        flaws.append("binary_clinical_risk")

    # Flaw 3: Debt maturity before next catalyst (if data available)
    # This requires debt_maturity_date and next_catalyst_date comparison
    debt_maturity = fin_data.get("debt_maturity_months")
    days_to_cat = None

    if hasattr(cat_data, 'days_to_nearest_catalyst'):
        days_to_cat = cat_data.days_to_nearest_catalyst
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        days_to_cat = scores.get("days_to_nearest_catalyst")

    if debt_maturity is not None and days_to_cat is not None:
        debt_maturity_dec = _to_decimal(debt_maturity)
        days_to_cat_months = Decimal(str(days_to_cat)) / Decimal("30") if days_to_cat else None

        if debt_maturity_dec is not None and days_to_cat_months is not None:
            if debt_maturity_dec < days_to_cat_months:
                flaws.append("debt_before_catalyst")

    return flaws


def apply_existential_cap(
    score: Decimal,
    flaws: List[str],
) -> Tuple[Decimal, List[str]]:
    """
    Apply existential flaw cap to score.

    Args:
        score: Pre-cap composite score
        flaws: List of existential flaw identifiers

    Returns:
        Tuple of (capped_score, flags)
    """
    flags = []
    if flaws:
        cap = EXISTENTIAL_FLAW_CONFIG["existential_cap"]
        if score > cap:
            score = cap
            flags.append("existential_capped")
        # Add individual flaw flags
        for flaw in flaws:
            flags.append(f"existential_{flaw}")

    return score, flags


# =============================================================================
# ENHANCEMENT 3: CONFIDENCE-WEIGHTED AGGREGATION
# =============================================================================

def compute_confidence_weighted_contribution(
    normalized_score: Decimal,
    weight: Decimal,
    confidence: Decimal,
    confidence_floor: Decimal = None,
) -> Tuple[Decimal, Decimal]:
    """
    Compute contribution with confidence as a binding factor.

    The effective score is reduced based on confidence level, making
    low-confidence signals contribute less to the final score.

    Formula:
        conf_factor = confidence_floor + (1 - confidence_floor) * confidence
        effective_score = score * conf_factor
        contribution = effective_score * weight

    Args:
        normalized_score: Normalized component score (0-100)
        weight: Effective weight for this component
        confidence: Confidence level (0-1)
        confidence_floor: Minimum confidence factor (default from config)

    Returns:
        Tuple of (contribution, conf_factor)
    """
    if confidence_floor is None:
        confidence_floor = CONFIDENCE_WEIGHTED_CONFIG["confidence_floor"]

    # Confidence factor ranges from confidence_floor (low conf) to 1.0 (high conf)
    conf_factor = confidence_floor + (Decimal("1") - confidence_floor) * confidence
    conf_factor = _clamp(conf_factor, confidence_floor, Decimal("1.0"))

    effective_score = normalized_score * conf_factor
    contribution = effective_score * weight

    return _quantize_score(contribution), conf_factor


# =============================================================================
# ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
# =============================================================================

def apply_dynamic_ceilings(
    score: Decimal,
    lead_phase: Optional[str],
    days_to_catalyst: Optional[int],
    revenue_growth: Optional[Decimal],
    is_commercial: bool,
) -> Tuple[Decimal, List[str]]:
    """
    Apply dynamic score ceilings based on stage, catalyst timing, and commercial status.

    Ceilings prevent scores from being too high in structurally disadvantaged situations:
    - Early-stage companies (preclinical, phase 1) have lower ceilings
    - No near-term catalyst means limited upside
    - Commercial companies with declining revenue face ceiling

    Args:
        score: Pre-ceiling composite score
        lead_phase: Lead program phase (e.g., "preclinical", "phase_1", "phase_2")
        days_to_catalyst: Days until next catalyst (None if no catalyst)
        revenue_growth: YoY revenue growth rate (negative = decline)
        is_commercial: Whether company is revenue-generating commercial stage

    Returns:
        Tuple of (capped_score, ceilings_applied)
    """
    ceilings_applied = []
    result = score

    # Normalize lead_phase for matching
    phase_lower = lead_phase.lower() if lead_phase else ""

    # Stage ceiling for early-stage companies
    for phase_key, cap in STAGE_CEILING_CONFIG.items():
        if phase_key in phase_lower:
            if result > cap:
                result = cap
                ceilings_applied.append(f"stage_ceiling_{phase_key.replace(' ', '_')}")
            break  # Only apply one stage ceiling

    # Catalyst ceiling - no near-term catalyst limits upside
    if days_to_catalyst is None or days_to_catalyst > 365:
        cap = CATALYST_CEILING_CONFIG["no_catalyst_12mo"]
        if result > cap:
            result = cap
            ceilings_applied.append("no_catalyst_12mo_ceiling")
    elif days_to_catalyst > 180:
        cap = CATALYST_CEILING_CONFIG["no_catalyst_6mo"]
        if result > cap:
            result = cap
            ceilings_applied.append("no_catalyst_6mo_ceiling")

    # Commercial revenue declining ceiling
    if is_commercial and revenue_growth is not None and revenue_growth < Decimal("0"):
        cap = COMMERCIAL_CEILING_CONFIG["revenue_declining"]
        if result > cap:
            result = cap
            ceilings_applied.append("commercial_revenue_declining_ceiling")

    return _quantize_score(result), ceilings_applied


# =============================================================================
# ENHANCEMENT 5: ASYMMETRIC TRANSFORM (CONVEX DOWNSIDE, CONCAVE UPSIDE)
# =============================================================================

def apply_asymmetric_transform(
    normalized_score: Decimal,
    component: str,
) -> Decimal:
    """
    Apply asymmetric transformation to a normalized score.

    This transformation creates:
    - Concave upside: Positive deviations from neutral are dampened (+10 → +6)
    - Convex downside: Negative deviations are amplified using a quadratic curve
      that is anchored at both the floor (5) and neutral (50), preventing the
      "effective floor lift" bug where linear amplification would push scores
      below 12.5 to the floor.

    The quadratic downside ensures:
    - f(50) = 50 (neutral maps to neutral)
    - f(5) = 5 (floor maps to floor - preserves P5 resolution)
    - slope at 50 = downside_amplification (1.2x emphasis near center)

    This reflects the asymmetric nature of biotech investing where downside risks
    are often more severe than upside surprises, while preserving resolution in
    the lower tail of the distribution.

    Args:
        normalized_score: Normalized component score (0-100)
        component: Component name (for potential future component-specific transforms)

    Returns:
        Transformed score (0-100)
    """
    config = ASYMMETRY_CONFIG
    neutral = config["neutral_threshold"]
    floor = Decimal("5")
    ceiling = Decimal("95")

    delta = normalized_score - neutral

    if delta > 0:
        # Concave upside: dampen positive deviations (linear)
        transformed_score = neutral + delta * config["upside_dampening"]
    elif delta < 0:
        # Convex downside: quadratic anchored at floor and neutral
        # f(x) = center + b*d + a*d^2, where d = x - center
        # b = downside_amplification (slope at center)
        # a = (b - 1) / span, where span = center - floor
        # This ensures f(floor) = floor and f(center) = center
        span = neutral - floor  # 45
        b = config["downside_amplification"]  # 1.2
        a = (b - Decimal("1")) / span  # (1.2 - 1) / 45 = 0.00444...
        transformed_score = neutral + (b * delta) + (a * delta * delta)
    else:
        transformed_score = neutral

    # Clamp to valid range - use [5, 95] floor/ceiling to match winsorization output
    return _clamp(_quantize_score(transformed_score), floor, ceiling)


def apply_asymmetric_transform_to_contribution(
    contribution_delta: Decimal,
    component: str,
) -> Tuple[Decimal, List[str]]:
    """
    Apply asymmetric transform to contribution delta from neutral.

    Args:
        contribution_delta: Deviation of contribution from neutral
        component: Component name

    Returns:
        Tuple of (transformed_delta, flags)
    """
    flags = []
    config = ASYMMETRY_CONFIG

    if contribution_delta > 0:
        # Concave upside: dampen positive contributions
        transformed = contribution_delta * config["upside_dampening"]
        if abs(transformed - contribution_delta) > Decimal("0.5"):
            flags.append("asymmetric_upside_dampened")
    elif contribution_delta < 0:
        # Convex downside: amplify negative contributions
        transformed = contribution_delta * config["downside_amplification"]
        if abs(transformed - contribution_delta) > Decimal("0.5"):
            flags.append("asymmetric_downside_amplified")
    else:
        transformed = Decimal("0")

    return _quantize_score(transformed), flags


# =============================================================================
# ENHANCEMENT 11: SMART MONEY REINFORCEMENT (OPTION D) - BAKER-STYLE
# =============================================================================

def compute_smart_money_reinforcement(
    stage_bucket: str,
    tier1_overlap: int,
    smart_money_confidence: Decimal,
    smart_money_trend: Optional[str],  # "increasing", "decreasing", "stable", None
    mode: "ScoringMode",
    conviction_overlap: Optional[float] = None,  # NEW: conviction-weighted overlap
    tier1_conviction_overlap: Optional[float] = None,  # NEW: T1-only conviction
    days_to_catalyst: Optional[int] = None,  # NEW: for timing calculation
    cohort_conviction_stats: Optional[Dict[str, float]] = None,  # NEW: for z-scoring
    thesis_gate_triggered: bool = False,  # NEW: block reinforcement if thesis gate fired
) -> Tuple[Decimal, Decimal, List[str], Dict[str, Any]]:
    """
    Compute smart-money reinforcement multipliers for catalyst and momentum.

    BAKER-STYLE CONVICTION × TIMING:
    In PoC/Regulatory stages, conviction-weighted overlap acts as an interaction
    coefficient scaled by catalyst proximity:

      strength = sigmoid(z_conviction)  # 0..1 based on cohort z-score
      timing = exp(-((days_to_cat - 90)/90)^2)  # peaks ~90d, relevant 30-180d
      reinforce = strength * timing  # 0..1

      catalyst_mult = 1 + 0.10 * reinforce
      momentum_mult = 1 + 0.06 * reinforce

    Falls back to tier1_overlap-based logic if conviction data unavailable.

    Args:
        stage_bucket: Development stage ('early', 'poc', 'pivotal', 'regulatory', 'commercial')
        tier1_overlap: Number of Tier-1 funds holding the position (fallback)
        smart_money_confidence: Confidence in smart money signal (0-1)
        smart_money_trend: Direction of position changes
        mode: Scoring mode
        conviction_overlap: Baker-style conviction-weighted overlap (preferred)
        tier1_conviction_overlap: Tier-1 only conviction overlap
        days_to_catalyst: Days until nearest catalyst (for timing)
        cohort_conviction_stats: {"mean": float, "std": float} for z-scoring

    Returns:
        Tuple of (catalyst_mult, momentum_mult, flags, diagnostics)
    """
    import math

    config = SMART_MONEY_REINFORCEMENT_CONFIG
    flags = []
    diagnostics = {
        "reinforcement_enabled": config["enabled"],
        "reinforcement_applied": False,
        "stage_bucket": stage_bucket,
        "tier1_overlap": tier1_overlap,
        "smart_money_confidence": str(smart_money_confidence),
        "conviction_overlap": conviction_overlap,
        "tier1_conviction_overlap": tier1_conviction_overlap,
        "days_to_catalyst": days_to_catalyst,
    }

    # Default: no reinforcement
    catalyst_mult = Decimal("1.00")
    momentum_mult = Decimal("1.00")

    # Only apply in BAKER_STYLE mode
    if mode != ScoringMode.BAKER_STYLE:
        diagnostics["skip_reason"] = "not_baker_style"
        return catalyst_mult, momentum_mult, flags, diagnostics

    if not config["enabled"]:
        diagnostics["skip_reason"] = "disabled"
        return catalyst_mult, momentum_mult, flags, diagnostics

    # THESIS GATE COUPLING: Don't reinforce weak-thesis names
    # This prevents smart money + timing from undoing Baker-style premise
    if thesis_gate_triggered:
        diagnostics["skip_reason"] = "thesis_gate_triggered"
        diagnostics["thesis_gate_blocked"] = True
        flags.append("sm_reinforcement_blocked_by_thesis_gate")
        return catalyst_mult, momentum_mult, flags, diagnostics

    # Only apply in event-alpha regimes (PoC/Regulatory)
    if stage_bucket not in config["active_stages"]:
        diagnostics["skip_reason"] = f"stage_{stage_bucket}_not_active"
        return catalyst_mult, momentum_mult, flags, diagnostics

    # Check for negative signal (selling) - always applies
    is_negative = smart_money_trend == "decreasing"
    if is_negative:
        cat_config = config["catalyst_reinforcement"]
        mom_config = config["momentum_reinforcement"]
        catalyst_mult = cat_config["negative_signal_mult"]
        momentum_mult = mom_config["negative_signal_mult"]
        flags.append("sm_reinforcement_negative")
        diagnostics["reinforcement_type"] = "negative"
        diagnostics["reinforcement_applied"] = True
        diagnostics["catalyst_mult"] = str(catalyst_mult)
        diagnostics["momentum_mult"] = str(momentum_mult)
        return catalyst_mult, momentum_mult, flags, diagnostics

    # =========================================================================
    # CONVICTION × TIMING REINFORCEMENT (Baker-style)
    # =========================================================================
    use_conviction = conviction_overlap is not None and conviction_overlap > 0

    if use_conviction:
        # Compute conviction strength via z-score + sigmoid
        if cohort_conviction_stats and cohort_conviction_stats.get("std", 0) > 0.01:
            mean = cohort_conviction_stats["mean"]
            std = cohort_conviction_stats["std"]
            z_conviction = (conviction_overlap - mean) / std
        else:
            # Fallback: use raw conviction scaled to reasonable range
            # Assume conviction_overlap of ~3.0 is "average"
            z_conviction = (conviction_overlap - 3.0) / 1.5

        # Sigmoid transform: z=0 → 0.5, z=2 → 0.88, z=-2 → 0.12
        strength = 1.0 / (1.0 + math.exp(-z_conviction))

        # Timing: bell-shaped around 90 days to catalyst
        # NO CATALYST = NO TIMING BOOST (PIT-safe, prevents "general momentum" reinforcement)
        if days_to_catalyst is not None and days_to_catalyst > 0:
            # Peak at ~90 days, drops off outside 30-180 range
            timing = math.exp(-((days_to_catalyst - 90) / 90) ** 2)
        else:
            # No catalyst data: no timing-based reinforcement (and do not mark as applied)
            diagnostics["timing_reason"] = "no_catalyst_data"
            diagnostics["skip_reason"] = "no_catalyst_data"
            diagnostics["reinforcement_applied"] = False
            return catalyst_mult, momentum_mult, flags, diagnostics

        # Combined reinforcement factor (0..1)
        reinforce = strength * timing
        reinforce = max(0.0, min(1.0, reinforce))

        # If reinforce is effectively zero, treat as not applied (avoids flag-only noise)
        if reinforce <= 0.0:
            diagnostics["skip_reason"] = "reinforce_zero"
            diagnostics["reinforcement_applied"] = False
            return catalyst_mult, momentum_mult, flags, diagnostics

        # Apply to multipliers
        # Max boost: +10% catalyst, +6% momentum when reinforce=1.0
        catalyst_mult = Decimal("1.00") + Decimal(str(round(0.10 * reinforce, 4)))
        momentum_mult = Decimal("1.00") + Decimal(str(round(0.06 * reinforce, 4)))

        diagnostics["conviction_method"] = "conviction_x_timing"
        diagnostics["z_conviction"] = round(z_conviction, 3)
        diagnostics["strength"] = round(strength, 3)
        diagnostics["timing"] = round(timing, 3)
        diagnostics["reinforce"] = round(reinforce, 3)

        if reinforce > 0.5:
            flags.append("sm_reinforcement_strong")
            diagnostics["reinforcement_type"] = "strong"
        elif reinforce > 0.25:
            flags.append("sm_reinforcement_moderate")
            diagnostics["reinforcement_type"] = "moderate"
        else:
            flags.append("sm_reinforcement_weak")
            diagnostics["reinforcement_type"] = "weak"

    else:
        # =====================================================================
        # FALLBACK: Tier-1 overlap-based (original logic)
        # Also requires a catalyst (consistent with conviction path)
        # =====================================================================

        # NO CATALYST = NO REINFORCEMENT (even in fallback)
        if days_to_catalyst is None or days_to_catalyst <= 0:
            diagnostics["skip_reason"] = "no_catalyst_for_fallback"
            return catalyst_mult, momentum_mult, flags, diagnostics

        min_conf = config["min_smart_money_confidence"]
        min_overlap = config["min_tier1_overlap"]

        if smart_money_confidence < min_conf and tier1_overlap < min_overlap:
            diagnostics["skip_reason"] = "below_threshold"
            return catalyst_mult, momentum_mult, flags, diagnostics

        cat_config = config["catalyst_reinforcement"]
        mom_config = config["momentum_reinforcement"]

        if tier1_overlap >= config["strong_overlap_threshold"]:
            catalyst_mult = cat_config["strong_overlap_mult"]
            momentum_mult = mom_config["strong_overlap_mult"]
            flags.append("sm_reinforcement_strong")
            diagnostics["reinforcement_type"] = "strong"
        elif tier1_overlap >= config["moderate_overlap_threshold"]:
            catalyst_mult = cat_config["moderate_overlap_mult"]
            momentum_mult = mom_config["moderate_overlap_mult"]
            flags.append("sm_reinforcement_moderate")
            diagnostics["reinforcement_type"] = "moderate"
        else:
            diagnostics["skip_reason"] = "overlap_too_low"
            return catalyst_mult, momentum_mult, flags, diagnostics

        diagnostics["conviction_method"] = "tier1_overlap_fallback"

    diagnostics["reinforcement_applied"] = True
    diagnostics["catalyst_mult"] = str(catalyst_mult)
    diagnostics["momentum_mult"] = str(momentum_mult)

    return catalyst_mult, momentum_mult, flags, diagnostics


# =============================================================================
# ENHANCEMENT 10: THESIS GATE (CLINICAL+POS ELIGIBILITY)
# =============================================================================

def apply_thesis_gate(
    score: Decimal,
    clinical_normalized: Optional[Decimal],
    pos_normalized: Optional[Decimal],
    stage_bucket: str,
    mode: "ScoringMode",
) -> Tuple[Decimal, List[str], Dict[str, Any]]:
    """
    Apply thesis gate to prevent weak-thesis names from ranking top.

    "You can't rank top without a real thesis" - Baker Bros style.
    If (clinical + pos) / 2 < stage threshold, cap the final score.

    Args:
        score: Pre-gate final score
        clinical_normalized: Normalized clinical score (0-100)
        pos_normalized: Normalized PoS score (0-100)
        stage_bucket: Development stage bucket
        mode: Scoring mode

    Returns:
        Tuple of (gated_score, flags, diagnostics)
    """
    config = THESIS_GATE_CONFIG
    flags = []
    diagnostics = {
        "thesis_gate_enabled": config["enabled"],
        "thesis_gate_triggered": False,
        "thesis_score": None,
        "thesis_threshold": None,
        "thesis_ceiling": None,
        "mode_is_baker": mode == ScoringMode.BAKER_STYLE,
    }

    # Only apply thesis gate in BAKER_STYLE mode
    if mode != ScoringMode.BAKER_STYLE:
        return score, flags, diagnostics

    if not config["enabled"]:
        return score, flags, diagnostics

    # Need both clinical and pos to compute thesis score
    if clinical_normalized is None or pos_normalized is None:
        flags.append("thesis_gate_skipped_missing_data")
        diagnostics["thesis_gate_skipped"] = "missing_data"
        return score, flags, diagnostics

    # Compute thesis score: average of clinical + pos
    thesis_score = (clinical_normalized + pos_normalized) / Decimal("2")
    thesis_score = _quantize_score(thesis_score)

    # Get threshold and ceiling for this stage
    threshold = config["thresholds_by_stage"].get(
        stage_bucket, config["thresholds_by_stage"]["none"]
    )
    ceiling = config["ceiling_by_stage"].get(
        stage_bucket, config["ceiling_by_stage"]["none"]
    )

    diagnostics["thesis_score"] = str(thesis_score)
    diagnostics["thesis_threshold"] = str(threshold)
    diagnostics["thesis_ceiling"] = str(ceiling)
    diagnostics["stage_bucket"] = stage_bucket

    # Check if thesis is weak
    if thesis_score < threshold:
        diagnostics["thesis_gate_triggered"] = True

        if config["use_soft_penalty"]:
            # Soft penalty: multiplicative reduction
            penalty_base = config["soft_penalty_base"]
            penalty_floor = config["soft_penalty_floor"]
            # Penalty scales with how far below threshold
            gap = threshold - thesis_score
            gap_factor = min(gap / Decimal("20"), Decimal("1"))  # Max at 20pt gap
            penalty = max(
                penalty_floor,
                penalty_base - gap_factor * (penalty_base - penalty_floor)
            )
            gated_score = score * penalty
            flags.append(f"thesis_gate_penalty_{penalty:.2f}")
            diagnostics["penalty_factor"] = str(penalty)
        else:
            # Hard ceiling: cap the score
            if score > ceiling:
                gated_score = ceiling
                flags.append(f"thesis_gate_ceiling_{stage_bucket}")
                diagnostics["score_capped_from"] = str(score)
            else:
                gated_score = score
                flags.append("thesis_gate_below_ceiling")

        return _quantize_score(gated_score), flags, diagnostics

    # Thesis is strong enough, no gating
    flags.append("thesis_gate_passed")
    return score, flags, diagnostics


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def _score_single_ticker_v3(
    ticker: str,
    fin_data: Dict,
    cat_data: Any,
    clin_data: Dict,
    pos_data: Optional[Dict],
    si_data: Optional[Dict],
    market_data: Optional[Dict],
    coinvest_data: Dict,
    base_weights: Dict[str, Decimal],
    regime: str,
    mode: ScoringMode,
    normalized_scores: Dict[str, Optional[Decimal]],
    cohort_key: str,
    normalization_method: NormalizationMethod,
    peer_valuations: List[Dict],
    fda_data: Optional[Dict] = None,
    diversity_data: Optional[Dict] = None,
    intensity_data: Optional[Dict] = None,
    partnership_data: Optional[Dict] = None,
    cash_burn_data: Optional[Dict] = None,
    phase_momentum_data: Optional[Dict] = None,
    cohort_overlap_stats: Optional[Dict[Tuple[str, str], Tuple[Decimal, Decimal]]] = None,
    cohort_conviction_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score a single ticker with all v3 enhancements."""

    flags = []

    # Extract raw scores
    fin_raw = _to_decimal(_coalesce(fin_data.get("financial_normalized"), fin_data.get("financial_score")))
    clin_raw = _to_decimal(clin_data.get("clinical_score"))
    pos_raw = _to_decimal(pos_data.get("pos_score")) if pos_data else None

    # Extract short interest score
    si_raw = _to_decimal(si_data.get("short_signal_score")) if si_data else None
    si_squeeze_potential = si_data.get("squeeze_potential", "UNKNOWN") if si_data else "UNKNOWN"
    si_crowding_risk = si_data.get("crowding_risk", "UNKNOWN") if si_data else "UNKNOWN"
    si_signal_direction = si_data.get("signal_direction", "NEUTRAL") if si_data else "NEUTRAL"
    si_trend_direction = si_data.get("trend_direction", "UNKNOWN") if si_data else "UNKNOWN"

    # Extract FDA designation data
    fda_designation_score = _to_decimal(fda_data.get("designation_score")) if fda_data else None
    fda_pos_multiplier = _to_decimal(fda_data.get("pos_multiplier")) if fda_data else Decimal("1.0")
    fda_timeline_acceleration = fda_data.get("timeline_acceleration_months", 0) if fda_data else 0
    fda_designation_types = fda_data.get("designation_types", []) if fda_data else []

    # Apply FDA designation multiplier to PoS score
    # BTD (+25%), Fast Track (+12%), etc. boost approval probability
    pos_raw_unadjusted = pos_raw
    if pos_raw is not None and fda_pos_multiplier and fda_pos_multiplier > Decimal("1.0"):
        pos_raw = (pos_raw * fda_pos_multiplier).quantize(Decimal("0.01"))
        # Cap at 95 to avoid overconfidence
        pos_raw = min(pos_raw, Decimal("95"))

    # Extract pipeline diversity data
    diversity_score = _to_decimal(diversity_data.get("diversity_score")) if diversity_data else None
    diversity_risk_profile = diversity_data.get("risk_profile", "unknown") if diversity_data else "unknown"
    diversity_program_count = diversity_data.get("program_count", 0) if diversity_data else 0
    diversity_platform_validated = diversity_data.get("platform_validated", False) if diversity_data else False

    # Extract competitive intensity data
    intensity_score = _to_decimal(intensity_data.get("competitive_intensity_score")) if intensity_data else None
    intensity_crowding = intensity_data.get("crowding_level", "unknown") if intensity_data else "unknown"
    intensity_position = intensity_data.get("competitive_position", "unknown") if intensity_data else "unknown"
    intensity_competitor_count = intensity_data.get("competitor_count", 0) if intensity_data else 0
    intensity_has_approved = intensity_data.get("has_approved_competition", False) if intensity_data else False

    # Extract partnership validation data
    partnership_score = _to_decimal(partnership_data.get("partnership_score")) if partnership_data else None
    partnership_strength = partnership_data.get("partnership_strength", "unknown") if partnership_data else "unknown"
    partnership_count = partnership_data.get("partnership_count", 0) if partnership_data else 0
    partnership_top_tier = partnership_data.get("top_tier_partners", 0) if partnership_data else 0
    partnership_total_value = _to_decimal(partnership_data.get("total_deal_value", 0)) if partnership_data else Decimal("0")
    partnership_top_partners = partnership_data.get("top_partners", []) if partnership_data else []

    # Extract cash burn trajectory data
    cash_burn_trajectory = cash_burn_data.get("trajectory", "unknown") if cash_burn_data else "unknown"
    cash_burn_risk = cash_burn_data.get("risk_level", "unknown") if cash_burn_data else "unknown"

    # Extract phase momentum data
    phase_momentum_value = phase_momentum_data.get("momentum", "unknown") if phase_momentum_data else "unknown"
    phase_momentum_confidence = _to_decimal(phase_momentum_data.get("confidence", 0)) if phase_momentum_data else Decimal("0")
    phase_momentum_lead_phase = phase_momentum_data.get("current_lead_phase", "unknown") if phase_momentum_data else "unknown"

    # Extract catalyst scores and metadata
    if hasattr(cat_data, 'score_blended'):
        cat_window = _to_decimal(cat_data.score_blended)
        cat_proximity = _to_decimal(getattr(cat_data, 'catalyst_proximity_score', 0)) or Decimal("0")
        cat_delta = _to_decimal(getattr(cat_data, 'catalyst_delta_score', 0)) or Decimal("0")
        days_to_cat = _coalesce(
            getattr(cat_data, 'catalyst_window_days', None),
            getattr(cat_data, 'days_to_nearest_catalyst', None),
        )
        cat_event_type = getattr(cat_data, 'nearest_catalyst_type', "DEFAULT")
    elif isinstance(cat_data, dict):
        scores = cat_data.get("scores", cat_data)
        cat_window = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        cat_proximity = _to_decimal(scores.get("catalyst_proximity_score", 0)) or Decimal("0")
        cat_delta = _to_decimal(scores.get("catalyst_delta_score", 0)) or Decimal("0")
        integration = cat_data.get("integration", {})
        days_to_cat = _coalesce(
            integration.get("catalyst_window_days"),
            scores.get("days_to_nearest_catalyst"),
            scores.get("catalyst_window_days"),
        )
        cat_event_type = scores.get("nearest_catalyst_type", "DEFAULT")
    else:
        cat_window = None
        cat_proximity = Decimal("0")
        cat_delta = Decimal("0")
        days_to_cat = None
        cat_event_type = "DEFAULT"

    # Compute effective catalyst score
    cat_effective, cat_proximity_blended, cat_blend_mode = _compute_catalyst_effective(
        cat_window, cat_proximity
    )
    cat_raw = cat_effective
    if cat_proximity_blended:
        flags.append("catalyst_proximity_blended")

    # PDUFA date presence signals FILED stage (NDA submitted)
    # FILED stage has 85% base rate vs Phase 3's 50% - apply adjustment
    # This captures companies that CT.gov still shows as Phase 3 but have filed NDA
    pdufa_pos_boost_applied = False
    pdufa_lead_phase = clin_data.get("lead_phase") if clin_data else None
    if cat_event_type == "FDA_PDUFA_DATE" and pos_raw is not None:
        # If lead_phase is Phase 3 but PDUFA exists, apply FILED stage adjustment
        # FILED base rate (85%) vs Phase 3 (42-50%) = ~1.7x multiplier
        if pdufa_lead_phase and "3" in pdufa_lead_phase.lower() and pos_raw < Decimal("70"):
            pdufa_multiplier = Decimal("1.70")  # Adjust Phase 3 → FILED equivalent
            pos_raw = (pos_raw * pdufa_multiplier).quantize(Decimal("0.01"))
            pos_raw = min(pos_raw, Decimal("95"))  # Cap at 95
            pdufa_pos_boost_applied = True
            flags.append("pdufa_filed_stage_boost")

    # Get normalized scores
    clin_norm = _coalesce(normalized_scores.get("clinical"), clin_raw, default=Decimal("50"))
    fin_norm = _coalesce(normalized_scores.get("financial"), fin_raw, default=Decimal("50"))
    cat_norm = _coalesce(normalized_scores.get("catalyst"), cat_raw, default=Decimal("50"))
    pos_norm = _coalesce(normalized_scores.get("pos"), pos_raw, default=Decimal("0"))

    # Extract financial metadata
    runway_months = _to_decimal(fin_data.get("runway_months"))
    liquidity_status = fin_data.get("liquidity_gate_status")
    dilution_bucket = fin_data.get("dilution_risk_bucket")
    market_cap_mm = _to_decimal(fin_data.get("market_cap_mm"))

    # V2: Extract financial data for valuation regime routing
    has_revenue = fin_data.get("has_revenue", False)
    revenue_mm = None
    cfo_mm = None
    enterprise_value_mm = None

    # Extract actual CFO and Revenue from financial data (raw values in dollars)
    raw_cfo = fin_data.get("CFO")
    raw_revenue = fin_data.get("Revenue")

    # Convert to millions
    if raw_revenue is not None:
        revenue_mm = _to_decimal(raw_revenue) / Decimal("1000000")
    elif fin_data.get("revenue_scale_bucket") in ("medium", "large", "mega"):
        # Fallback: Use market_cap as proxy if exact revenue not available
        revenue_mm = market_cap_mm

    if raw_cfo is not None:
        cfo_mm = _to_decimal(raw_cfo) / Decimal("1000000")
    elif fin_data.get("burn_source") == "profitable":
        # Fallback: Profitable = positive CFO, use liquid_assets as proxy
        cfo_mm = _to_decimal(fin_data.get("liquid_assets")) / Decimal("1000000") if fin_data.get("liquid_assets") else Decimal("1")
    elif fin_data.get("monthly_burn") == 0 or fin_data.get("monthly_burn") == 0.0:
        # Fallback: No burn = likely profitable
        cfo_mm = Decimal("1")

    # Enterprise value from market data if available
    if market_data:
        enterprise_value_mm = _to_decimal(market_data.get("enterprise_value"))
        if enterprise_value_mm:
            enterprise_value_mm = enterprise_value_mm / Decimal("1000000")  # Convert to millions

    # Get stage and trial count
    lead_phase = clin_data.get("lead_phase")
    stage = _stage_bucket(lead_phase)
    trial_count = clin_data.get("trial_count", 0)
    phase_counts = clin_data.get("phase_counts", {})

    # Extract market data
    annualized_vol = None
    momentum_input = MultiWindowMomentumInput()
    momentum_source = "prices"

    if market_data:
        vol_val = market_data.get("volatility_252d")
        annualized_vol = _to_decimal(vol_val if vol_val is not None else market_data.get("annualized_volatility"))

        if market_data.get("_momentum_source") == "13f":
            momentum_source = "13f"

        # Extract benchmark values with None-safe fallback (0.0 is a valid value)
        # CRITICAL: Do NOT use `val or fallback` pattern - 0.0 is falsy but valid!
        _xbi_20 = market_data.get("xbi_return_20d")
        _xbi_60 = market_data.get("xbi_return_60d")
        _xbi_120 = market_data.get("xbi_return_120d")

        momentum_input = MultiWindowMomentumInput(
            return_20d=_to_decimal(market_data.get("return_20d")),
            return_60d=_to_decimal(market_data.get("return_60d")),
            return_120d=_to_decimal(market_data.get("return_120d")),
            benchmark_20d=_to_decimal(_xbi_20 if _xbi_20 is not None else market_data.get("benchmark_return_20d")),
            benchmark_60d=_to_decimal(_xbi_60 if _xbi_60 is not None else market_data.get("benchmark_return_60d")),
            benchmark_120d=_to_decimal(_xbi_120 if _xbi_120 is not None else market_data.get("benchmark_return_120d")),
            trading_days_available=market_data.get("trading_days_available"),
            annualized_vol=annualized_vol,
        )

    # =========================================================================
    # COMPUTE ALL ENHANCEMENTS
    # =========================================================================

    # 1. Volatility adjustment
    vol_adj = compute_volatility_adjustment(annualized_vol)
    if vol_adj.vol_bucket == VolatilityBucket.HIGH:
        flags.append("high_volatility")
    elif vol_adj.vol_bucket == VolatilityBucket.LOW:
        flags.append("low_volatility_boost")

    # 2. Momentum signal (multi-window blended)
    # Uses weighted blend of 20d (20%), 60d (50%), 120d (30%) for robust signal
    momentum = compute_momentum_signal_multiwindow(
        momentum_input,
        use_vol_adjusted_alpha=False,
        blend_mode="weighted",  # Use multi-window blend instead of fallback
    )
    momentum_norm = momentum.momentum_score

    if momentum.data_status == "missing_prices":
        flags.append("momentum_missing_prices")
    elif momentum.data_status == "computed_low_conf":
        flags.append("momentum_low_confidence")

    if momentum.alpha_60d is not None:
        if momentum.alpha_60d >= Decimal("0.10"):
            flags.append("strong_positive_momentum")
        elif momentum.alpha_60d <= Decimal("-0.10"):
            flags.append("strong_negative_momentum")

    if momentum.window_used:
        flags.append(f"momentum_window_{momentum.window_used}d")

    if momentum.data_completeness < Decimal("0.5"):
        flags.append("momentum_data_incomplete")

    # 3. Valuation signal (V2: with regime routing)
    valuation = compute_valuation_signal(
        market_cap_mm, trial_count, lead_phase, peer_valuations,
        revenue_mm=revenue_mm,
        cfo_mm=cfo_mm,
        enterprise_value_mm=enterprise_value_mm,
        has_revenue=has_revenue,
    )
    valuation_norm = valuation.valuation_score
    if valuation.valuation_score >= Decimal("70"):
        flags.append("undervalued_vs_peers")
    elif valuation.valuation_score <= Decimal("30"):
        flags.append("overvalued_vs_peers")

    # Add valuation regime and method flags
    if valuation.regime:
        flags.append(f"valuation_regime_{valuation.regime.value}")
    if valuation.flags:
        flags.extend(valuation.flags)

    # 4. Catalyst decay
    decay = compute_catalyst_decay(days_to_cat, cat_event_type or "DEFAULT")
    if decay.in_optimal_window:
        flags.append("catalyst_optimal_window")

    cat_norm_decayed = apply_catalyst_decay(cat_norm, decay)

    # 5. Smart money signal
    smart_money = compute_smart_money_signal(
        coinvest_data.get("coinvest_overlap_count", 0),
        coinvest_data.get("coinvest_holders", []),
        coinvest_data.get("position_changes"),
        coinvest_data.get("holder_tiers"),
    )
    if smart_money.holders_increasing:
        flags.append("smart_money_buying")
    if smart_money.holders_decreasing:
        flags.append("smart_money_selling")
    if smart_money.tier1_holders:
        flags.append("smart_money_tier1_present")

    # 6. Interaction terms
    if runway_months is None:
        runway_gate = "UNKNOWN"
    elif runway_months < Decimal("6"):
        runway_gate = "FAIL"
    elif runway_months >= Decimal("12"):
        runway_gate = "PASS"
    else:
        runway_gate = "UNKNOWN"
    dilution_gate = "PASS" if dilution_bucket in ("LOW", "MEDIUM") else (
        "FAIL" if dilution_bucket in ("HIGH", "SEVERE") else "UNKNOWN"
    )

    interactions = compute_interaction_terms(
        clin_norm,
        fin_data,
        cat_norm,
        stage,
        vol_adj,
        runway_gate_status=runway_gate,
        dilution_gate_status=dilution_gate,
        competitive_crowding=intensity_crowding,
        partnership_strength=partnership_strength,
        cash_burn_trajectory=cash_burn_trajectory,
        cash_burn_risk=cash_burn_risk,
        phase_momentum=phase_momentum_value,
        phase_momentum_confidence=phase_momentum_confidence,
        phase_momentum_lead_phase=phase_momentum_lead_phase,
        days_to_nearest_catalyst=days_to_cat,
    )
    flags.extend(interactions.interaction_flags)

    # 7. Financial Module 2: Survivability & Capital Discipline
    # Bounded additive term [-10, +5] with 0.06 weight
    survivability_result = compute_survivability_score(fin_data)
    survivability_score = Decimal(str(survivability_result.get("score", 0)))
    survivability_subscores = survivability_result.get("subscores", {})
    survivability_coverage = survivability_result.get("coverage", [])

    if survivability_score < Decimal("-3"):
        flags.append("survivability_critical")
    elif survivability_score < Decimal("0"):
        flags.append("survivability_warning")
    elif survivability_score >= Decimal("3"):
        flags.append("survivability_strong")

    if "missing_cash" in survivability_coverage:
        flags.append("survivability_missing_cash")
    if "missing_burn_data" in survivability_coverage:
        flags.append("survivability_missing_burn")

    # =========================================================================
    # CONFIDENCE EXTRACTION
    # =========================================================================

    conf_clin = _extract_confidence_clinical(clin_data)
    conf_fin = _extract_confidence_financial(fin_data)
    conf_cat = _extract_confidence_catalyst(cat_data)
    conf_pos = _extract_confidence_pos(pos_data)
    conf_si = _extract_confidence_short_interest(si_data)

    conf_clin = _clamp(conf_clin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_fin = _clamp(conf_fin - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))
    conf_cat = _clamp(conf_cat - vol_adj.confidence_penalty, Decimal("0.1"), Decimal("1"))

    # =========================================================================
    # WEIGHT COMPUTATION WITH STAGE/SIZE TILTS
    # =========================================================================

    # 1. Determine stage bucket (event-aware for better tilting)
    stage_bucket_alpha = _determine_stage_bucket_alpha(
        lead_phase=lead_phase,
        cat_event_type=cat_event_type,
        days_to_catalyst=days_to_cat,
    )

    # 2. Determine size bucket
    size_bucket = _market_cap_bucket(market_cap_mm) if market_cap_mm else "mid"

    # 3. Apply stage/size tilts to base weights
    # Use Baker-style tilts for BAKER_STYLE mode, standard tilts otherwise
    if mode == ScoringMode.BAKER_STYLE:
        tilt_config = BAKER_STYLE_TILT_CONFIG
    else:
        tilt_config = STAGE_SIZE_TILT_CONFIG

    tilted_weights, tilt_diagnostics = _apply_stage_size_tilts(
        base_weights=base_weights,
        stage_bucket=stage_bucket_alpha,
        size_bucket=size_bucket,
        enable_tilting=tilt_config.get("enabled", True),
        tilt_config=tilt_config,
    )

    if tilt_diagnostics["tilt_applied"]:
        flags.append(f"stage_tilt_{stage_bucket_alpha}")
        flags.append(f"size_tilt_{size_bucket}")

    # 3.5 Enhanced smart money signal (A+B+C+E) - now with stage/size context
    enhanced_smart_money = None
    if SMART_MONEY_ENHANCEMENT_CONFIG.get("enabled", True):
        enhanced_smart_money = compute_enhanced_smart_money_signal_v2(
            coinvest_data=coinvest_data,
            stage_bucket=stage_bucket_alpha,
            size_bucket=size_bucket,
            cohort_stats=cohort_overlap_stats,
        )
        # Add enhanced smart money flags
        flags.extend(enhanced_smart_money.get("flags", []))

    # 4. Apply regime adjustments on top of tilted weights
    regime_weights = apply_regime_to_weights(tilted_weights, regime)

    # =========================================================================
    # ENHANCEMENT 1: HARD REGIME GATING
    # Apply regime-specific gates to normalized scores before contribution calc
    # =========================================================================

    pre_gate_scores = {
        "clinical": clin_norm,
        "financial": fin_norm,
        "catalyst": cat_norm,
        "momentum": momentum_norm,
        "valuation": valuation_norm,
    }

    gated_scores, regime_config, regime_gate_flags = apply_regime_gates(pre_gate_scores, regime)
    flags.extend(regime_gate_flags)

    # Update normalized scores with gated values
    if gated_scores.get("momentum") is not None and gated_scores["momentum"] != momentum_norm:
        momentum_norm = gated_scores["momentum"]
    if gated_scores.get("valuation") is not None and gated_scores["valuation"] != valuation_norm:
        valuation_norm = gated_scores["valuation"]

    # Store financial penalty multiplier for later use
    financial_penalty_mult = regime_config.get("financial_penalty_mult", Decimal("1.0"))

    vol_adjusted_weights = {
        k: v * vol_adj.weight_adjustment_factor
        for k, v in regime_weights.items()
    }

    confidences = {
        "clinical": conf_clin,
        "financial": conf_fin,
        "catalyst": conf_cat,
        "pos": conf_pos,
        "momentum": momentum.confidence,
        "valuation": valuation.confidence,
        "short_interest": conf_si,
    }

    gated_components = []

    effective_weights = {}
    for comp, base_w in vol_adjusted_weights.items():
        conf = confidences.get(comp, Decimal("0.5"))

        # Continuous confidence gate: smooth transition instead of hard cutoff
        # - At conf=0: gate_factor=0
        # - At conf=THRESHOLD: gate_factor=1
        # - Above threshold: gate_factor=1
        # This prevents high-quality but distant events from being zeroed out
        gate_factor = min(conf / CONFIDENCE_GATE_THRESHOLD, Decimal("1"))

        if gate_factor < Decimal("1"):
            # Partial gating: weight is reduced but not zeroed
            eff_w = base_w * gate_factor * (Decimal("0.5") + conf * Decimal("0.5"))
            gated_components.append(comp)
            flags.append(f"{comp}_confidence_soft_gated")
        else:
            # Full confidence: standard weight calculation
            eff_w = base_w * (Decimal("0.5") + conf * Decimal("0.5"))

        effective_weights[comp] = eff_w

    # =========================================================================
    # CONVICTION HORIZON TOLERANCE OVERLAY
    # =========================================================================
    # For high-conviction names with real catalysts in 90-240 day window,
    # provide an ADDITIVE boost to catalyst weight on top of the continuous gate.
    # This gives meaningful catalyst exposure for legitimate long-dated events.
    #
    # CRITICAL: thesis_gate remains a HARD BLOCK for this overlay.

    cat_base_w = vol_adjusted_weights.get("catalyst", Decimal("0"))
    cat_eff_w = effective_weights.get("catalyst", Decimal("0"))

    # Pre-compute thesis gate check for overlay (matches later full check)
    overlay_thesis_blocked = False
    if mode == ScoringMode.BAKER_STYLE and THESIS_GATE_CONFIG["enabled"]:
        if clin_norm is not None and pos_norm is not None:
            thesis_score_pre = (clin_norm + pos_norm) / Decimal("2")
            # Prefer already-computed display stage when present; fall back to lead_phase
            display_stage = (
                stage if stage in {"early", "mid", "late"}
                else _stage_bucket(lead_phase)
            )
            threshold_pre = THESIS_GATE_CONFIG["thresholds_by_stage"].get(
                display_stage, THESIS_GATE_CONFIG["thresholds_by_stage"]["none"]
            )
            overlay_thesis_blocked = thesis_score_pre < threshold_pre

    # Apply overlay via wrapper (handles all pre-conditions and diagnostics)
    new_cat_w, conviction_horizon_overlay = _apply_conviction_horizon_overlay(
        weight_base=cat_base_w,
        weight_effective=cat_eff_w,
        confidence_low=(conf_cat < CONFIDENCE_GATE_THRESHOLD),
        thesis_gate_blocked=overlay_thesis_blocked,
        conviction_overlap=coinvest_data.get("conviction_overlap") if coinvest_data else None,
        tier1_count=int(coinvest_data.get("tier1_count", 0) or 0) if coinvest_data else 0,
        days_to_catalyst=int(days_to_cat) if days_to_cat is not None else None,
        cohort_conviction_stats=cohort_conviction_stats,
    )
    effective_weights["catalyst"] = new_cat_w
    if conviction_horizon_overlay.get("applied"):
        flags.append("conviction_horizon_overlay_applied")

    total = sum(effective_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in effective_weights.items()}

    # =========================================================================
    # COVERAGE-GATED SMART MONEY WEIGHTING (ENHANCED A+B+C+E)
    # =========================================================================
    # If security has 13F coverage, include smart_money with dynamic weight
    # based on stage/size and tier-1 count. Uses enhanced signal if enabled.

    has_smart_money_coverage = smart_money.overlap_count > 0
    if has_smart_money_coverage:
        if enhanced_smart_money and SMART_MONEY_ENHANCEMENT_CONFIG.get("enabled", True):
            # Use dynamic weight from enhanced signal (A: stage×size dependent)
            effective_weights["smart_money"] = enhanced_smart_money["weight"]
            flags.append("smart_money_enhanced")
        else:
            # Fallback to static weight
            effective_weights["smart_money"] = SMART_MONEY_COVERAGE_GATED_WEIGHT
        flags.append("smart_money_coverage_included")
    else:
        effective_weights["smart_money"] = Decimal("0")
        flags.append("smart_money_no_coverage")

    # Renormalize weights to sum to 1.0 after adding smart_money
    total = sum(effective_weights.values())
    if total > EPS:
        effective_weights = {k: _quantize_weight(v / total) for k, v in effective_weights.items()}

    # =========================================================================
    # COMPOSITE SCORE COMPUTATION
    # =========================================================================

    component_scores = []
    contributions = {}
    confidence_factors = {}  # Track confidence factors for each component
    asymmetric_flags = []  # Track asymmetric transform flags

    core_scores = {
        "clinical": (clin_norm, clin_raw, conf_clin),
        "financial": (fin_norm, fin_raw, conf_fin),
        "catalyst": (cat_norm_decayed, cat_raw, conf_cat),
    }

    for name, (norm, raw, conf) in core_scores.items():
        w_eff = effective_weights.get(name, Decimal("0"))

        # ENHANCEMENT 5: Apply asymmetric transform to normalized score
        transformed_norm = apply_asymmetric_transform(norm, name)
        delta_from_transform = transformed_norm - norm
        if abs(delta_from_transform) > Decimal("0.5"):
            if delta_from_transform < 0:
                asymmetric_flags.append(f"{name}_asymmetric_upside_dampened")
            else:
                asymmetric_flags.append(f"{name}_asymmetric_downside_amplified")

        # ENHANCEMENT 3: Confidence-weighted contribution
        contrib, conf_factor = compute_confidence_weighted_contribution(
            transformed_norm, w_eff, conf
        )
        contributions[name] = contrib
        confidence_factors[name] = conf_factor

        notes = []
        if raw is None:
            notes.append("missing_raw")
        if name == "catalyst" and decay.decay_factor < Decimal("0.9"):
            notes.append(f"decay_factor_{decay.decay_factor}")
        if conf_factor < Decimal("0.7"):
            notes.append(f"conf_weighted_{conf_factor}")

        component_scores.append(ComponentScore(
            name=name,
            raw=raw,
            normalized=transformed_norm,  # Store transformed value
            confidence=conf,
            weight_base=base_weights.get(name, Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(contrib),
            decay_factor=decay.decay_factor if name == "catalyst" else None,
            notes=notes,
        ))

    # Add asymmetric transform flags
    flags.extend(asymmetric_flags)

    if mode in (ScoringMode.ENHANCED, ScoringMode.PARTIAL):
        if "momentum" in effective_weights:
            w_eff = effective_weights["momentum"]

            # ENHANCEMENT 5: Apply asymmetric transform
            transformed_momentum = apply_asymmetric_transform(momentum_norm, "momentum")
            delta_from_transform = transformed_momentum - momentum_norm
            if abs(delta_from_transform) > Decimal("0.5"):
                if delta_from_transform < 0:
                    flags.append("momentum_asymmetric_upside_dampened")
                else:
                    flags.append("momentum_asymmetric_downside_amplified")

            # ENHANCEMENT 3: Confidence-weighted contribution
            contrib, conf_factor = compute_confidence_weighted_contribution(
                transformed_momentum, w_eff, momentum.confidence
            )
            contributions["momentum"] = contrib
            confidence_factors["momentum"] = conf_factor

            mom_notes = [] if momentum.alpha_60d is not None else ["missing_price_data"]
            if conf_factor < Decimal("0.7"):
                mom_notes.append(f"conf_weighted_{conf_factor}")

            component_scores.append(ComponentScore(
                name="momentum",
                raw=momentum.alpha_60d,
                normalized=transformed_momentum,
                confidence=momentum.confidence,
                weight_base=base_weights.get("momentum", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=mom_notes,
            ))

        if "valuation" in effective_weights:
            w_eff = effective_weights["valuation"]

            # ENHANCEMENT 5: Apply asymmetric transform
            transformed_valuation = apply_asymmetric_transform(valuation_norm, "valuation")
            delta_from_transform = transformed_valuation - valuation_norm
            if abs(delta_from_transform) > Decimal("0.5"):
                if delta_from_transform < 0:
                    flags.append("valuation_asymmetric_upside_dampened")
                else:
                    flags.append("valuation_asymmetric_downside_amplified")

            # ENHANCEMENT 3: Confidence-weighted contribution
            contrib, conf_factor = compute_confidence_weighted_contribution(
                transformed_valuation, w_eff, valuation.confidence
            )
            contributions["valuation"] = contrib
            confidence_factors["valuation"] = conf_factor

            val_notes = [] if valuation.peer_count >= 5 else ["insufficient_peers"]
            if conf_factor < Decimal("0.7"):
                val_notes.append(f"conf_weighted_{conf_factor}")

            component_scores.append(ComponentScore(
                name="valuation",
                raw=valuation.mcap_per_asset,
                normalized=transformed_valuation,
                confidence=valuation.confidence,
                weight_base=base_weights.get("valuation", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=val_notes,
            ))

        # Short interest signal integration
        if "short_interest" in effective_weights and si_raw is not None:
            w_eff = effective_weights["short_interest"]
            # Short interest score is already 0-100, use directly as normalized
            si_norm = si_raw
            contrib = si_norm * w_eff
            contributions["short_interest"] = contrib

            si_notes = []
            if si_squeeze_potential == "EXTREME":
                flags.append("extreme_squeeze_potential")
                si_notes.append("extreme_squeeze")
            elif si_squeeze_potential == "HIGH":
                flags.append("high_squeeze_potential")
                si_notes.append("high_squeeze")
            if si_crowding_risk == "HIGH":
                flags.append("high_short_crowding")
                si_notes.append("high_crowding")
            if si_signal_direction == "BULLISH":
                flags.append("si_bullish_signal")
            elif si_signal_direction == "BEARISH":
                flags.append("si_bearish_signal")
            if si_trend_direction == "COVERING":
                flags.append("shorts_covering")
            elif si_trend_direction == "BUILDING":
                flags.append("shorts_building")

            component_scores.append(ComponentScore(
                name="short_interest",
                raw=si_raw,
                normalized=si_norm,
                confidence=conf_si,
                weight_base=base_weights.get("short_interest", Decimal("0")),
                weight_effective=w_eff,
                contribution=_quantize_score(contrib),
                notes=si_notes if si_notes else [],
            ))
            flags.append("short_interest_applied")

    pos_contrib_raw = Decimal("0")
    pos_contrib_capped = Decimal("0")
    pos_delta_was_capped = False

    if mode in (ScoringMode.ENHANCED, ScoringMode.BAKER_STYLE) and pos_raw is not None:
        w_eff = effective_weights.get("pos", Decimal("0"))
        pos_contrib_raw = pos_norm * w_eff

        pos_contrib_capped = _clamp(pos_contrib_raw, -POS_DELTA_CAP, POS_DELTA_CAP)

        if pos_contrib_raw != pos_contrib_capped:
            pos_delta_was_capped = True
            flags.append("pos_delta_capped")

        contributions["pos"] = pos_contrib_capped

        component_scores.append(ComponentScore(
            name="pos",
            raw=pos_raw,
            normalized=pos_norm,
            confidence=conf_pos,
            weight_base=base_weights.get("pos", Decimal("0")),
            weight_effective=w_eff,
            contribution=_quantize_score(pos_contrib_capped),
            notes=["delta_capped"] if pos_delta_was_capped else [],
        ))
        flags.append("pos_score_applied")

    # =========================================================================
    # SMART MONEY CONTRIBUTION (Enhanced A+B+C+E)
    # =========================================================================
    # Only add contribution if coverage exists (weight > 0 after gating)
    # Uses enhanced signal if available for expanded score range and confidence

    sm_w_eff = effective_weights.get("smart_money", Decimal("0"))
    if sm_w_eff > EPS and has_smart_money_coverage:
        # Use enhanced smart money signal if available
        if enhanced_smart_money and SMART_MONEY_ENHANCEMENT_CONFIG.get("enabled", True):
            # Enhanced signal: expanded range (20-85), cohort-normalized, confidence from overlap
            sm_norm = enhanced_smart_money["score"]
            sm_confidence = enhanced_smart_money["confidence"]
            sm_base_weight = enhanced_smart_money["weight"]
        else:
            # Fallback to original signal
            sm_norm = smart_money.smart_money_score
            sm_confidence = smart_money.confidence
            sm_base_weight = SMART_MONEY_COVERAGE_GATED_WEIGHT

        sm_contrib = sm_norm * sm_w_eff
        contributions["smart_money"] = sm_contrib

        sm_notes = []
        if enhanced_smart_money:
            if enhanced_smart_money.get("tier1_count", 0) > 0:
                sm_notes.append(f"tier1:{enhanced_smart_money['tier1_count']}")
            if enhanced_smart_money.get("cohort_z_score"):
                sm_notes.append(f"z:{enhanced_smart_money['cohort_z_score']}")
            if enhanced_smart_money.get("increasing_holders"):
                sm_notes.append(f"buying:{len(enhanced_smart_money['increasing_holders'])}")
            if enhanced_smart_money.get("decreasing_holders"):
                sm_notes.append(f"selling:{len(enhanced_smart_money['decreasing_holders'])}")
        else:
            if smart_money.holders_increasing:
                sm_notes.append(f"buying:{len(smart_money.holders_increasing)}")
            if smart_money.holders_decreasing:
                sm_notes.append(f"selling:{len(smart_money.holders_decreasing)}")
            if smart_money.tier1_holders:
                sm_notes.append(f"tier1:{len(smart_money.tier1_holders)}")
            if smart_money.conditional_capped:
                sm_notes.append("conditional_capped")

        component_scores.append(ComponentScore(
            name="smart_money",
            raw=Decimal(str(smart_money.overlap_count)),  # Raw overlap count
            normalized=sm_norm,
            confidence=sm_confidence,
            weight_base=sm_base_weight,
            weight_effective=sm_w_eff,
            contribution=_quantize_score(sm_contrib),
            notes=sm_notes if sm_notes else [],
        ))
        flags.append("smart_money_applied")

    # =========================================================================
    # ENHANCEMENT 11: SMART MONEY REINFORCEMENT (OPTION D) - BAKER-STYLE
    # Apply conviction × timing reinforcement in PoC/Regulatory stages
    # =========================================================================

    # Extract tier1 overlap and trend from coinvest data
    tier1_overlap = coinvest_data.get("coinvest_overlap_count", 0)
    if not tier1_overlap:
        # Try alternate key from enhanced smart money signal
        tier1_overlap = enhanced_smart_money.get("tier1_count", 0) if enhanced_smart_money else 0

    sm_trend = coinvest_data.get("position_trend")  # "increasing", "decreasing", "stable"

    # NEW: Extract conviction overlap (Baker-style) if available
    conviction_overlap = coinvest_data.get("conviction_overlap")
    tier1_conviction_overlap = coinvest_data.get("tier1_conviction_overlap")

    # Get sm_confidence (may not be defined if smart money wasn't applied)
    sm_confidence_for_reinforcement = Decimal("0.5")  # Default
    if sm_w_eff > EPS and has_smart_money_coverage:
        if enhanced_smart_money and SMART_MONEY_ENHANCEMENT_CONFIG.get("enabled", True):
            sm_confidence_for_reinforcement = enhanced_smart_money.get("confidence", Decimal("0.5"))
        else:
            sm_confidence_for_reinforcement = smart_money.confidence

    # Pre-check thesis gate for reinforcement coupling (Baker-style)
    # Don't allow reinforcement to boost weak-thesis names
    # FIX: Use display stage (early/mid/late) instead of alpha stage (poc/pivotal/etc)
    # This prevents late-stage companies with trial readouts from being treated as PoC
    thesis_gate_would_trigger = False
    thesis_gate_bypass_reason = None
    if mode == ScoringMode.BAKER_STYLE and THESIS_GATE_CONFIG["enabled"]:
        if clin_norm is not None and pos_norm is not None:
            thesis_score = (clin_norm + pos_norm) / Decimal("2")

            # Use display stage bucket for threshold (not alpha stage)
            # This allows late-stage companies to use the "late" threshold (45)
            # instead of being forced into "poc" threshold (55) by catalyst type
            # Prefer already-computed display stage when present; fall back to lead_phase
            display_stage_bucket = (
                stage if stage in {"early", "mid", "late"}
                else _stage_bucket(lead_phase)
            )
            threshold = THESIS_GATE_CONFIG["thresholds_by_stage"].get(
                display_stage_bucket, THESIS_GATE_CONFIG["thresholds_by_stage"]["none"]
            )

            thesis_gate_would_trigger = thesis_score < threshold

            # Conviction override: very high institutional signal can bypass thesis gate
            # This allows names like INSM (14 tier-1, conviction 23) to receive reinforcement
            conv_override = THESIS_GATE_CONFIG.get("conviction_override", {})
            if thesis_gate_would_trigger and conv_override.get("enabled", False):
                min_tier1 = conv_override.get("min_tier1", 10)
                min_conviction = conv_override.get("min_conviction", Decimal("15"))
                if (tier1_overlap or 0) >= min_tier1 and _to_decimal(conviction_overlap or 0) >= min_conviction:
                    thesis_gate_would_trigger = False
                    thesis_gate_bypass_reason = "conviction_override"
                    flags.append("thesis_gate_conviction_override")

    reinforcement_catalyst_mult, reinforcement_momentum_mult, reinforcement_flags, reinforcement_diagnostics = (
        compute_smart_money_reinforcement(
            stage_bucket=stage_bucket_alpha,
            tier1_overlap=tier1_overlap,
            smart_money_confidence=sm_confidence_for_reinforcement,
            smart_money_trend=sm_trend,
            mode=mode,
            # NEW: Conviction × timing parameters
            conviction_overlap=conviction_overlap,
            tier1_conviction_overlap=tier1_conviction_overlap,
            days_to_catalyst=days_to_cat,  # Already computed earlier in function
            cohort_conviction_stats=cohort_conviction_stats,  # Stage×size cohort stats for z-scoring
            thesis_gate_triggered=thesis_gate_would_trigger,  # Block reinforcement if thesis weak
        )
    )
    # Add thesis gate diagnostic details (display stage + bypass reason)
    # Prefer already-computed display stage when present; fall back to lead_phase
    reinforcement_diagnostics["thesis_display_stage"] = (
        stage if stage in {"early", "mid", "late"}
        else _stage_bucket(lead_phase)
    )
    if thesis_gate_bypass_reason:
        reinforcement_diagnostics["thesis_gate_bypass"] = thesis_gate_bypass_reason
    flags.extend(reinforcement_flags)

    # Apply reinforcement to contributions (not scores - keeps attribution clean)
    if reinforcement_diagnostics.get("reinforcement_applied"):
        if "catalyst" in contributions:
            original_cat = contributions["catalyst"]
            contributions["catalyst"] = original_cat * reinforcement_catalyst_mult
            reinforcement_diagnostics["catalyst_original"] = str(original_cat)
            reinforcement_diagnostics["catalyst_reinforced"] = str(contributions["catalyst"])

        if "momentum" in contributions:
            original_mom = contributions["momentum"]
            contributions["momentum"] = original_mom * reinforcement_momentum_mult
            reinforcement_diagnostics["momentum_original"] = str(original_mom)
            reinforcement_diagnostics["momentum_reinforced"] = str(contributions["momentum"])

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    weighted_sum = sum(contributions.values())

    critical_scores = []
    for c in CRITICAL_COMPONENTS:
        if c in contributions and c in effective_weights and effective_weights[c] > EPS:
            underlying = contributions[c] / effective_weights[c]
            critical_scores.append(underlying)

    min_critical = min(critical_scores) if critical_scores else weighted_sum

    pre_penalty = HYBRID_ALPHA * weighted_sum + (Decimal("1") - HYBRID_ALPHA) * min_critical

    pre_penalty = pre_penalty + interactions.total_interaction_adjustment
    pre_penalty = _quantize_score(pre_penalty)

    # =========================================================================
    # ENHANCEMENT 6: CONTRADICTION DETECTOR
    # Detect conflicting signals that reduce confidence in the score
    # =========================================================================

    contradiction_data = {
        "momentum_score": momentum_norm,
        "valuation_score": valuation_norm,
        "clinical_score": clin_norm,
        "financial_score": fin_norm,
        "liquidity_gate": liquidity_status,
        "runway_months": runway_months,
        "dilution_bucket": dilution_bucket,
    }

    contradictions = detect_contradictions(contradiction_data)

    if contradictions.contradictions:
        # Apply contradiction score penalty
        pre_penalty = pre_penalty - contradictions.score_penalty
        pre_penalty = _quantize_score(max(pre_penalty, Decimal("0")))

        # Add contradiction flags
        for contradiction in contradictions.contradictions:
            flags.append(f"contradiction_{contradiction}")

    # =========================================================================
    # PENALTIES AND CAPS
    # =========================================================================

    subfactors = [clin_raw, fin_raw, cat_raw]
    if mode == ScoringMode.ENHANCED:
        subfactors.append(pos_raw)
    missing_count = sum(1 for x in subfactors if x is None)
    missing_pct = Decimal(str(missing_count / len(subfactors)))
    uncertainty_penalty = min(MAX_UNCERTAINTY_PENALTY, missing_pct)

    post_uncertainty = pre_penalty * (Decimal("1") - uncertainty_penalty)

    severities = [fin_data.get("severity", "none"), clin_data.get("severity", "none")]
    if hasattr(cat_data, 'severe_negative_flag') and cat_data.severe_negative_flag:
        severities.append("sev1")
    elif isinstance(cat_data, dict):
        flags_dict = cat_data.get("flags", {})
        if isinstance(flags_dict, dict) and flags_dict.get("severe_negative_flag"):
            severities.append("sev1")
    worst_severity = _get_worst_severity(severities)
    severity_multiplier = SEVERITY_MULTIPLIERS[worst_severity]

    post_severity = post_uncertainty * severity_multiplier
    post_severity = _quantize_score(post_severity)

    post_cap, caps_applied = _apply_monotonic_caps(
        post_severity,
        liquidity_status,
        runway_months,
        dilution_bucket
    )

    # =========================================================================
    # ENHANCEMENT 4: DYNAMIC SCORE CEILINGS
    # Apply stage, catalyst, and commercial revenue ceilings
    # =========================================================================

    # Determine if commercial (has revenue)
    is_commercial = has_revenue or (revenue_mm is not None and revenue_mm > Decimal("0"))

    # Calculate revenue growth if available
    revenue_growth = None
    if fin_data.get("revenue_yoy_growth") is not None:
        revenue_growth = _to_decimal(fin_data.get("revenue_yoy_growth"))
    elif fin_data.get("revenue_growth_rate") is not None:
        revenue_growth = _to_decimal(fin_data.get("revenue_growth_rate"))

    post_ceiling, ceilings_applied = apply_dynamic_ceilings(
        post_cap,
        lead_phase,
        days_to_cat,
        revenue_growth,
        is_commercial,
    )

    if ceilings_applied:
        flags.extend(ceilings_applied)
        caps_applied.extend([{"reason": c, "cap": str(post_ceiling)} for c in ceilings_applied])

    # =========================================================================
    # ENHANCEMENT 2: EXISTENTIAL FLAW CAPS
    # Detect and cap for existential risks (runway, binary clinical, debt)
    # =========================================================================

    existential_flaws = detect_existential_flaws(fin_data, clin_data, cat_data)
    post_existential, existential_flags = apply_existential_cap(post_ceiling, existential_flaws)

    if existential_flags:
        flags.extend(existential_flags)
        if post_existential < post_ceiling:
            caps_applied.append({
                "reason": "existential_cap",
                "cap": str(EXISTENTIAL_FLAW_CONFIG["existential_cap"]),
                "flaws": existential_flaws,
            })

    post_vol = apply_volatility_to_score(post_existential, vol_adj)

    delta_bonus = (cat_delta / Decimal("25")).quantize(SCORE_PRECISION)

    # Financial Module 2: Survivability contribution (bounded additive term)
    # Weight 0.06, score range [-10, +5] -> contribution range [-0.6, +0.3]
    SURVIVABILITY_WEIGHT = Decimal("0.06")
    survivability_contribution = (survivability_score * SURVIVABILITY_WEIGHT).quantize(SCORE_PRECISION)

    # Allow negative raw scores (e.g. negative-weight models) without collapsing to 0
    final_score = _clamp(post_vol + delta_bonus + survivability_contribution, Decimal("-100"), Decimal("100"))
    final_score = _quantize_score(final_score)

    # =========================================================================
    # ENHANCEMENT 10: THESIS GATE (BAKER_STYLE mode only)
    # Prevent weak-thesis names from ranking top
    # =========================================================================

    # Use the display stage bucket (early/mid/late) for thesis gate threshold lookup,
    # NOT stage_bucket_alpha. This allows late-stage companies (Phase 3+) to use the
    # "late" threshold (45) instead of being forced into "poc" threshold (55) by
    # their catalyst event type. See lines 3242-3244 for the same logic in the
    # smart money reinforcement pre-check.
    final_score, thesis_gate_flags, thesis_gate_diagnostics = apply_thesis_gate(
        score=final_score,
        clinical_normalized=clin_norm,
        pos_normalized=pos_norm,
        stage_bucket=stage,  # Use display stage (early/mid/late), not alpha stage
        mode=mode,
    )
    flags.extend(thesis_gate_flags)

    # =========================================================================
    # BUILD OUTPUT
    # =========================================================================

    if uncertainty_penalty > 0:
        flags.append("uncertainty_penalty_applied")
    if worst_severity == Severity.SEV2:
        flags.append("sev2_penalty_applied")
    elif worst_severity == Severity.SEV1:
        flags.append("sev1_penalty_applied")
    if caps_applied:
        flags.append("monotonic_cap_applied")

    enhancements_dict = {
        "momentum_score": momentum.momentum_score,
        "valuation_score": valuation.valuation_score,
        "smart_money_score": smart_money.smart_money_score,
        "vol_bucket": vol_adj.vol_bucket.value,
        "catalyst_decay": decay.decay_factor,
        "interaction_adjustment": interactions.total_interaction_adjustment,
        "survivability_score": survivability_score,
        "survivability_contribution": survivability_contribution,
        # New enhancement tracking
        "regime_gates_applied": bool(regime_gate_flags),
        "contradiction_penalty": contradictions.score_penalty,
        "contradictions_detected": contradictions.contradictions,
        "ceilings_applied": ceilings_applied,
        "existential_flaws": existential_flaws,
        "confidence_factors": {k: str(v) for k, v in confidence_factors.items()},
        "stage_size_weighting": tilt_diagnostics,
        # Enhancement 10: Thesis gate (Baker-style)
        "thesis_gate": thesis_gate_diagnostics,
        # Enhancement 11: Smart money reinforcement (Option D)
        "smart_money_reinforcement": reinforcement_diagnostics,
        # Enhancement 12: Conviction horizon tolerance overlay
        "conviction_horizon_overlay": conviction_horizon_overlay,
    }

    determinism_hash = _compute_determinism_hash(
        ticker, SCHEMA_VERSION, mode.value, base_weights, effective_weights,
        component_scores, enhancements_dict, final_score
    )

    confidence_overall = (
        conf_clin * Decimal("0.3") +
        conf_fin * Decimal("0.3") +
        conf_cat * Decimal("0.2") +
        momentum.confidence * Decimal("0.1") +
        valuation.confidence * Decimal("0.1")
    )
    confidence_overall = _clamp(confidence_overall, Decimal("0.1"), Decimal("0.9"))

    breakdown = ScoreBreakdown(
        version=SCHEMA_VERSION,
        mode=mode.value,
        base_weights={k: str(v) for k, v in base_weights.items()},
        regime_adjustments={"regime": regime},
        effective_weights={k: str(v) for k, v in effective_weights.items()},
        components=[{
            "name": c.name, "raw": str(c.raw) if c.raw is not None else None,
            "normalized": str(c.normalized) if c.normalized is not None else None,
            "confidence": str(c.confidence), "weight_base": str(c.weight_base),
            "weight_effective": str(c.weight_effective), "contribution": str(c.contribution),
            "decay_factor": str(c.decay_factor) if c.decay_factor is not None else None,
            "notes": c.notes,
        } for c in component_scores],
        enhancements={
            "momentum": {"score": str(momentum.momentum_score), "alpha_60d": str(momentum.alpha_60d) if momentum.alpha_60d else None, "confidence": str(momentum.confidence)},
            "valuation": {"score": str(valuation.valuation_score), "peer_count": valuation.peer_count, "confidence": str(valuation.confidence)},
            "smart_money": {"score": str(smart_money.smart_money_score), "overlap_count": smart_money.overlap_count},
            "volatility": {"bucket": vol_adj.vol_bucket.value, "weight_factor": str(vol_adj.weight_adjustment_factor), "score_factor": str(vol_adj.score_adjustment_factor)},
            "catalyst_decay": {"factor": str(decay.decay_factor), "in_optimal_window": decay.in_optimal_window},
            "confidence_factors": {k: str(v) for k, v in confidence_factors.items()},
            "stage_size_weighting": tilt_diagnostics,
            # Baker-style enhancements (Enhancement 10 & 11)
            "thesis_gate": thesis_gate_diagnostics,
            "smart_money_reinforcement": reinforcement_diagnostics,
            # Enhancement 12: Conviction horizon tolerance overlay
            "conviction_horizon_overlay": conviction_horizon_overlay,
        },
        penalties_and_gates={
            "uncertainty_penalty_pct": str(_quantize_score(uncertainty_penalty * Decimal("100"))),
            "severity_gate": SEVERITY_GATE_LABELS[worst_severity],
            "severity_multiplier": str(severity_multiplier),
            "monotonic_caps_applied": caps_applied,
            # New enhancement penalties
            "contradiction_penalty": str(contradictions.score_penalty),
            "contradictions": contradictions.contradictions,
            "dynamic_ceilings": ceilings_applied,
            "existential_flaws": existential_flaws,
            "regime_gates": regime_gate_flags,
        },
        interaction_terms={
            "clinical_financial_synergy": str(interactions.clinical_financial_synergy),
            "stage_financial_interaction": str(interactions.stage_financial_interaction),
            "catalyst_volatility_dampening": str(interactions.catalyst_volatility_dampening),
            "total_adjustment": str(interactions.total_interaction_adjustment),
            "flags": interactions.interaction_flags,
        },
        final={
            "pre_penalty_score": str(pre_penalty),
            "post_contradiction_score": str(_quantize_score(pre_penalty - contradictions.score_penalty)),
            "post_uncertainty_score": str(_quantize_score(post_uncertainty)),
            "post_severity_score": str(post_severity),
            "post_cap_score": str(post_cap),
            "post_ceiling_score": str(post_ceiling),
            "post_existential_score": str(post_existential),
            "post_vol_score": str(post_vol),
            "delta_bonus": str(delta_bonus),
            "composite_score": str(final_score),
        },
        normalization_method=normalization_method.value,
        cohort_info={"cohort_key": cohort_key},
        hybrid_aggregation={
            "alpha": str(HYBRID_ALPHA),
            "weighted_sum": str(_quantize_score(weighted_sum)),
            "min_critical": str(_quantize_score(min_critical)),
        },
    )

    # Phase-weighted trial computation (diagnostic only, no scoring impact)
    _pw_weights = {
        "phase_1": 0.5, "phase_1_2": 0.75, "phase_2": 1.0,
        "phase_2_3": 1.5, "phase_3": 2.0, "approved": 3.0, "other": 0.25,
    }
    phase_weighted_trials = sum(phase_counts.get(p, 0) * w for p, w in _pw_weights.items())
    mcap_per_pw_trial = (
        float(market_cap_mm) / phase_weighted_trials
        if phase_weighted_trials > 0 and market_cap_mm is not None
        else None
    )

    # Valuation applicability and notes (diagnostic only)
    _val_size_bucket = _market_cap_bucket(market_cap_mm)
    _val_notes_parts = []
    if market_cap_mm is None:
        _val_notes_parts.append("missing_mcap")
    if trial_count < 2:
        _val_notes_parts.append(f"denom<2 (trial_count={trial_count})")
    if not phase_counts:
        _val_notes_parts.append("missing_phase_breakdown")
    if valuation.flags:
        _val_notes_parts.extend(valuation.flags)
    _val_applicable = trial_count >= 2 and market_cap_mm is not None

    return {
        "ticker": ticker,
        "composite_score": final_score,
        "severity": worst_severity,
        "flags": sorted(set(flags)),
        "determinism_hash": determinism_hash,
        "score_breakdown": breakdown,
        "confidence_clinical": conf_clin,
        "confidence_financial": conf_fin,
        "confidence_catalyst": conf_cat,
        "confidence_pos": conf_pos if mode in (ScoringMode.ENHANCED, ScoringMode.BAKER_STYLE, ScoringMode.ADAPTIVE) else None,
        "confidence_short_interest": conf_si if si_raw is not None else None,
        "confidence_overall": confidence_overall,
        "effective_weights": effective_weights,
        "normalization_method": normalization_method.value,
        "caps_applied": caps_applied,
        # NOTE: component_scores kept as raw dataclass list for internal use.
        # Serialization happens in module_5_composite_v3.py via score_breakdown.components
        # which is the single source of truth for JSON output.
        "component_scores": component_scores,
        "uncertainty_penalty": uncertainty_penalty,
        "momentum_signal": {
            "momentum_score": str(momentum.momentum_score),
            "alpha_60d": str(momentum.alpha_60d) if momentum.alpha_60d else None,
            "confidence": str(momentum.confidence),
            "data_completeness": str(momentum.data_completeness),
            "window_used": momentum.window_used,
            "data_status": momentum.data_status,
            "source": momentum_source,
        },
        "valuation_signal": {
            "valuation_score": str(valuation.valuation_score),
            "peer_count": valuation.peer_count,
            "confidence": str(valuation.confidence),
            # V2 regime routing fields
            "regime": valuation.regime.value if hasattr(valuation.regime, 'value') else str(valuation.regime),
            "method": valuation.method,
            "ev_multiple": str(valuation.ev_multiple) if valuation.ev_multiple else None,
            "peer_median_ev_multiple": str(valuation.peer_median_ev_multiple) if valuation.peer_median_ev_multiple else None,
            "mcap_per_asset": str(valuation.mcap_per_asset) if valuation.mcap_per_asset else None,
            "peer_median_mcap_per_asset": str(valuation.peer_median_mcap_per_asset) if valuation.peer_median_mcap_per_asset else None,
            "flags": valuation.flags if valuation.flags else [],
            # Debug/audit fields (ranking-neutral)
            "valuation_raw_metric": (
                str(valuation.mcap_per_asset) if valuation.mcap_per_asset is not None
                else str(valuation.ev_multiple) if valuation.ev_multiple is not None
                else None
            ),
            "valuation_mcap_mm": str(valuation.mcap_used_mm) if valuation.mcap_used_mm is not None else None,
            "valuation_denom_trials_total": trial_count,
            "valuation_denom_trials_p1": phase_counts.get("phase_1", 0) + phase_counts.get("phase_1_2", 0),
            "valuation_denom_trials_p2": phase_counts.get("phase_2", 0) + phase_counts.get("phase_2_3", 0),
            "valuation_denom_trials_p3": phase_counts.get("phase_3", 0),
            "valuation_denom_trials_reg": phase_counts.get("approved", 0),
            "valuation_denom_trials_other": phase_counts.get("other", 0),
            "valuation_phase_weighted_trials": round(phase_weighted_trials, 2) if phase_weighted_trials else None,
            "valuation_metric_pw": round(mcap_per_pw_trial, 2) if mcap_per_pw_trial else None,
            "valuation_size_bucket": _val_size_bucket,
            "valuation_applicable": _val_applicable,
            "valuation_notes": "; ".join(_val_notes_parts) if _val_notes_parts else "",
        },
        "smart_money_signal": {
            "score": str(smart_money.smart_money_score),
            "overlap_count": smart_money.overlap_count,
            "weighted_overlap": str(smart_money.weighted_overlap),
            "overlap_bonus": str(smart_money.overlap_bonus),
            "change_adjustment": str(smart_money.position_change_adjustment),
            "confidence": str(smart_money.confidence),
            "tier_breakdown": smart_money.tier_breakdown,
            "tier1_holders": smart_money.tier1_holders,
            "holders_increasing": smart_money.holders_increasing,
            "holders_decreasing": smart_money.holders_decreasing,
        },
        "short_interest_signal": {
            "score": str(si_raw) if si_raw is not None else None,
            "squeeze_potential": si_squeeze_potential,
            "crowding_risk": si_crowding_risk,
            "signal_direction": si_signal_direction,
            "trend_direction": si_trend_direction,
            "confidence": str(conf_si),
            "applied": si_raw is not None and "short_interest" in effective_weights,
        },
        "volatility_adjustment": {
            "annualized_vol_pct": str(vol_adj.annualized_vol) if vol_adj.annualized_vol else None,
            "vol_bucket": vol_adj.vol_bucket.value,
            "weight_factor": str(vol_adj.weight_adjustment_factor),
            "score_factor": str(vol_adj.score_adjustment_factor),
        },
        "catalyst_decay": {
            "factor": str(decay.decay_factor),
            "days_to_catalyst": decay.days_to_catalyst,
            "in_optimal_window": decay.in_optimal_window,
        },
        "catalyst_effective": {
            "catalyst_score_window": str(cat_window) if cat_window is not None else None,
            "catalyst_proximity_score": str(cat_proximity) if cat_proximity else "0",
            "catalyst_score_effective": str(cat_effective),
            "catalyst_proximity_blended": cat_proximity_blended,
            "blend_mode": cat_blend_mode,
        },
        "fda_designation_signal": {
            "designation_score": str(fda_designation_score) if fda_designation_score else None,
            "pos_multiplier": str(fda_pos_multiplier),
            "pos_unadjusted": str(pos_raw_unadjusted) if pos_raw_unadjusted else None,
            "pos_adjusted": str(pos_raw) if pos_raw else None,
            "pdufa_filed_stage_boost": pdufa_pos_boost_applied,
            "timeline_acceleration_months": fda_timeline_acceleration,
            "designation_types": fda_designation_types,
            "has_designations": len(fda_designation_types) > 0,
        },
        "pipeline_diversity_signal": {
            "diversity_score": str(diversity_score) if diversity_score else None,
            "risk_profile": diversity_risk_profile,
            "program_count": diversity_program_count,
            "platform_validated": diversity_platform_validated,
        },
        "competitive_intensity_signal": {
            "intensity_score": str(intensity_score) if intensity_score else None,
            "crowding_level": intensity_crowding,
            "competitive_position": intensity_position,
            "competitor_count": intensity_competitor_count,
            "has_approved_competition": intensity_has_approved,
        },
        "partnership_signal": {
            "partnership_score": str(partnership_score) if partnership_score else None,
            "partnership_strength": partnership_strength,
            "partnership_count": partnership_count,
            "top_tier_partners": partnership_top_tier,
            "total_deal_value_mm": str(partnership_total_value),
            "top_partners": partnership_top_partners,
        },
        "survivability_signal": {
            "score": str(survivability_score),
            "contribution": str(survivability_contribution),
            "subscores": {k: str(v) for k, v in survivability_subscores.items()},
            "coverage": survivability_coverage,
            "metrics": survivability_result.get("metrics", {}),
        },
        "interaction_terms": {
            "total_adjustment": str(interactions.total_interaction_adjustment),
            "flags": interactions.interaction_flags,
        },
        # New composite scoring enhancements tracking
        "regime_gating": {
            "regime": regime,
            "gates_applied": regime_gate_flags,
            "financial_penalty_mult": str(financial_penalty_mult),
        },
        "contradiction_detection": {
            "contradictions": contradictions.contradictions,
            "confidence_penalty": str(contradictions.confidence_penalty),
            "score_penalty": str(contradictions.score_penalty),
            "diagnostics": contradictions.diagnostics,
        },
        "dynamic_ceilings": {
            "ceilings_applied": ceilings_applied,
            "lead_phase": lead_phase,
            "days_to_catalyst": days_to_cat,
            "is_commercial": is_commercial,
            "revenue_growth": str(revenue_growth) if revenue_growth is not None else None,
        },
        "existential_flaw_detection": {
            "flaws": existential_flaws,
            "cap_applied": "existential_capped" in existential_flags,
            "cap_value": str(EXISTENTIAL_FLAW_CONFIG["existential_cap"]) if existential_flaws else None,
        },
        "confidence_weighting": {
            "factors": {k: str(v) for k, v in confidence_factors.items()},
            "floor": str(CONFIDENCE_WEIGHTED_CONFIG["confidence_floor"]),
        },
        "asymmetric_transform": {
            "upside_dampening": str(ASYMMETRY_CONFIG["upside_dampening"]),
            "downside_amplification": str(ASYMMETRY_CONFIG["downside_amplification"]),
            "flags_triggered": [f for f in flags if "asymmetric" in f],
        },
    }
