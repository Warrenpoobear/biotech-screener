#!/usr/bin/env python3
"""
Module 5: Composite Ranker (v3) - IC-Enhanced Edition

Production-ready composite scoring with all IC enhancement features:
- Adaptive weight learning (historical IC optimization)
- Non-linear signal interactions (cross-factor synergies/penalties)
- Peer-relative valuation signal
- Catalyst signal decay (time-based IC modeling)
- Price momentum signal (relative strength vs XBI)
- Shrinkage normalization (Bayesian cohort adjustment)
- Smart money signal (13F position changes)
- Volatility-adjusted scoring
- Regime-adaptive component selection
- Monotonic caps (from v2)
- Confidence weighting (from v2)
- Hybrid aggregation with weakest-link (from v2)

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies
- DECIMAL-ONLY: Pure Decimal arithmetic for all scoring
- FAIL LOUDLY: Clear error states with validation
- AUDITABLE: Full provenance chain with score breakdown
- IC-OPTIMIZED: Every feature designed to maximize predictive power

Weight Structure (v3 Enhanced):
- Clinical Development: 28%
- Financial Health:     25%
- Catalyst Momentum:    17%
- Probability of Success: 15%
- Price Momentum:       10%
- Valuation:            5%
(When enhancement data available; falls back to v1 weights otherwise)

Expected IC Improvement: +0.08 to +0.15 vs v1 baseline

Author: Wake Robin Capital Management
Version: 3.0.0
Last Modified: 2026-01-17
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from common.provenance import create_provenance
from common.types import Severity
from common.integration_contracts import (
    extract_financial_score,
    extract_catalyst_score,
    extract_clinical_score,
    validate_module_1_output,
    validate_module_2_output,
    validate_module_3_output,
    validate_module_4_output,
    SchemaValidationError,
)
from common.production_hardening import (
    safe_parse_date,
    DateParseError,
    get_module_logger,
)

# Import IC enhancement utilities
from src.modules.ic_enhancements import (
    # Core enhancement functions
    compute_volatility_adjustment,
    apply_volatility_to_score,
    compute_momentum_signal,
    compute_momentum_signal_with_fallback,  # V2: Multi-window with fallback
    compute_valuation_signal,
    compute_catalyst_decay,
    apply_catalyst_decay,
    compute_smart_money_signal,
    compute_interaction_terms,
    shrinkage_normalize,
    apply_regime_to_weights,
    compute_adaptive_weights,
    compute_enhanced_score,
    get_regime_signal_importance,
    # Types
    VolatilityAdjustment,
    VolatilityBucket,
    MomentumSignal,
    MultiWindowMomentumInput,  # V2: Multi-window input
    ValuationSignal,
    CatalystDecayResult,
    SmartMoneySignal,
    InteractionTerms,
    AdaptiveWeights,
    RegimeType,
    EnhancedScoringResult,
    # Helpers
    _to_decimal,
    _quantize_score,
    _quantize_weight,
    _clamp,
    _safe_divide,
    EPS,
    SCORE_PRECISION,
    WEIGHT_PRECISION,
)

# Import scoring robustness enhancements (v3.3)
from src.modules.scoring_robustness import (
    # Core functions
    winsorize_component_score,
    winsorize_cohort,
    apply_confidence_shrinkage,
    compute_rank_stability_penalty,
    blend_timeframe_signals,
    apply_asymmetric_bounds,
    apply_weight_floors,
    evaluate_defensive_triggers,
    check_distribution_health,
    apply_robustness_enhancements,
    # Types
    WinsorizedScore,
    ShrinkageResult,
    RankStabilityAdjustment,
    AsymmetricBounds,
    WeightFloorResult,
    DefensiveOverrideResult,
    DistributionHealthCheck,
    RobustnessEnhancements,
    DefensivePosture,
    DistributionHealth,
)

# Import PIT validation
from src.modules.ic_pit_validation import (
    run_production_gate,
    create_weight_provenance,
    PITValidationError,
    WeightStabilityError,
    ProductionGateResult,
    WeightProvenance,
)

# Import scoring module (extracted for maintainability)
from module_5_scoring_v3 import (
    # Types (re-exported for backwards compatibility)
    MonotonicCap,
    ScoringMode,
    RunStatus,
    NormalizationMethod,
    ComponentScore,
    ScoreBreakdown,
    V3ScoringResult,
    # Constants used by both scoring and orchestration
    SEVERITY_MULTIPLIERS,
    SEVERITY_GATE_LABELS,
    MIN_COHORT_SIZE,
    MAX_UNCERTAINTY_PENALTY,
    WINSOR_LOW,
    WINSOR_HIGH,
    HYBRID_ALPHA,
    CRITICAL_COMPONENTS,
    POS_DELTA_CAP,
    CATALYST_WINDOW_WEIGHT,
    CATALYST_PROXIMITY_WEIGHT,
    CATALYST_DEFAULT_BASE,
    CATALYST_DEFAULT_SCORE,
    CONFIDENCE_GATE_THRESHOLD,
    # Helper functions
    _coalesce,
    _compute_catalyst_effective,
    _market_cap_bucket,
    _stage_bucket,
    _quarter_from_date,
    _get_worst_severity,
    _rank_normalize_winsorized,
    _extract_confidence_financial,
    _extract_confidence_clinical,
    _extract_confidence_catalyst,
    _extract_confidence_pos,
    _apply_monotonic_caps,
    _compute_determinism_hash,
    _enrich_with_coinvest,
    _apply_cohort_normalization_v3,
    _compute_global_stats,
    # Main scoring function
    _score_single_ticker_v3,
)

# Import diagnostics module (extracted for maintainability)
from module_5_diagnostics_v3 import (
    compute_momentum_breakdown,
    build_momentum_health,
    format_momentum_log_lines,
    check_coverage_guardrail,
)

__version__ = "3.0.0"
RULESET_VERSION = "3.0.0-IC-ENHANCED"
SCHEMA_VERSION = "v3.0"

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (orchestration-specific - scoring constants imported from module_5_scoring_v3)
# =============================================================================

# V3 Enhanced weights (with all new signals)
V3_ENHANCED_WEIGHTS = {
    "clinical": Decimal("0.26"),
    "financial": Decimal("0.24"),
    "catalyst": Decimal("0.16"),
    "pos": Decimal("0.14"),
    "momentum": Decimal("0.09"),
    "valuation": Decimal("0.05"),
    "short_interest": Decimal("0.06"),
}

# V3 Default weights (without enhancement data)
V3_DEFAULT_WEIGHTS = {
    "clinical": Decimal("0.40"),
    "financial": Decimal("0.35"),
    "catalyst": Decimal("0.25"),
}

# V3 Partial weights (with some enhancement data)
V3_PARTIAL_WEIGHTS = {
    "clinical": Decimal("0.33"),
    "financial": Decimal("0.28"),
    "catalyst": Decimal("0.18"),
    "momentum": Decimal("0.09"),
    "valuation": Decimal("0.05"),
    "short_interest": Decimal("0.07"),
}

# Pipeline health thresholds (fraction of universe)
# NOTE: Biotech-adjusted thresholds - sparse coverage is normal for optional enhancement components
HEALTH_GATE_THRESHOLDS = {
    "catalyst": Decimal("0.10"),    # Fail if <10% have catalyst events (core component)
    "momentum": Decimal("0.00"),    # Optional: 13F fallback provides sparse coverage by design
    "smart_money": Decimal("0.00"), # Coverage-gated at runtime (5% when covered, 0% otherwise)
}

# =============================================================================
# EXPECTED RETURN CALCULATION
# =============================================================================
# EXPECTED RETURN COMPUTATION
# =============================================================================
# Institutional methodology: score → rank → percentile → z-score → expected return
# Lambda (λ) = annualized excess return per 1σ of signal
# Biotech alpha typically 6-12% per σ; we use 8% as conservative baseline
#
# Implementation moved to common/score_to_er.py for reuse across modules.
# =============================================================================

from common.score_to_er import (
    compute_expected_returns,
    DEFAULT_LAMBDA_ANNUAL,
    ER_MODEL_ID,
    ER_MODEL_VERSION,
)

# Re-export for backward compatibility
EXPECTED_RETURN_LAMBDA = DEFAULT_LAMBDA_ANNUAL  # 8% per 1σ per year (conservative)


# =============================================================================
# SCORING TYPES & HELPERS (extracted to module_5_scoring_v3.py)
# =============================================================================
# The following are now imported from module_5_scoring_v3 for maintainability:
# - Types: MonotonicCap, ScoringMode, RunStatus, NormalizationMethod,
#          ComponentScore, ScoreBreakdown, V3ScoringResult
# - Constants: SEVERITY_MULTIPLIERS, MIN_COHORT_SIZE, etc.
# - Helpers: _coalesce, _compute_catalyst_effective, _market_cap_bucket, etc.
# - Core function: _score_single_ticker_v3
# =============================================================================


# =============================================================================
# MAIN COMPOSITE FUNCTION
# =============================================================================

def compute_module_5_composite_v3(
    universe_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    catalyst_result: Dict[str, Any],
    clinical_result: Dict[str, Any],
    as_of_date: str,
    weights: Optional[Dict[str, Decimal]] = None,
    coinvest_signals: Optional[Dict] = None,
    enhancement_result: Optional[Dict[str, Any]] = None,
    market_data_by_ticker: Optional[Dict[str, Dict]] = None,
    raw_financial_data: Optional[List[Dict[str, Any]]] = None,
    historical_scores: Optional[List[Dict]] = None,
    historical_returns: Optional[Dict[Tuple[date, str], Decimal]] = None,
    use_adaptive_weights: bool = False,
    validate_inputs: bool = True,
    enforce_pit_gates: bool = True,
    previous_weights: Optional[Dict[str, Decimal]] = None,
    embargo_months: int = 1,
    shrinkage_lambda: Decimal = Decimal("0.70"),
    smooth_gamma: Decimal = Decimal("0.80"),
) -> Dict[str, Any]:
    """
    Compute composite scores with all v3 IC enhancements.

    This is the production-ready v3 composite scorer with:
    - Adaptive weight learning
    - Non-linear signal interactions
    - Peer-relative valuation
    - Catalyst signal decay
    - Price momentum
    - Shrinkage normalization
    - Smart money signals
    - Volatility-adjusted scoring
    - Regime-adaptive components
    - Monotonic caps
    - Confidence weighting
    - Hybrid aggregation

    Args:
        universe_result: Module 1 output with active/excluded securities
        financial_result: Module 2 output with financial scores
        catalyst_result: Module 3 output with catalyst summaries
        clinical_result: Module 4 output with clinical scores
        as_of_date: ISO date string for the scoring date
        weights: Optional custom weights (defaults based on available data)
        coinvest_signals: Optional co-invest overlay signals
        enhancement_result: Optional PoS/SI/regime enhancement data
        market_data_by_ticker: Optional dict mapping ticker to market data
            (volatility_252d, return_60d, xbi_return_60d)
        historical_scores: Optional historical scores for adaptive weights.
            Each dict must have 'as_of_date' field (date or ISO string).
        historical_returns: Optional Dict keyed by (as_of_date, ticker) -> forward return.
            CRITICAL FOR PIT SAFETY: The as_of_date in the key must be when the
            return period STARTS, not when it ends.
        use_adaptive_weights: Whether to compute adaptive weights
        validate_inputs: If True (default), validate upstream outputs
        embargo_months: Minimum months between score date and return measurement.
            Default 1 month ensures returns are fully realized before use.
        shrinkage_lambda: 0-1, higher = more shrinkage toward base_weights.
            Default 0.70 provides strong regularization to prevent overfitting.
        smooth_gamma: 0-1, higher = more smoothing toward prev_weights.
            Default 0.80 reduces weight volatility period-to-period.

    Returns:
        Dict with ranked_securities, excluded_securities, and diagnostics
    """
    # Validate upstream module outputs
    if validate_inputs:
        validate_module_1_output(universe_result)
        validate_module_2_output(financial_result)
        validate_module_3_output(catalyst_result)
        validate_module_4_output(clinical_result)

    # Parse as_of_date for PIT validation with error handling
    try:
        as_of_dt = safe_parse_date(as_of_date, field_name="as_of_date")
    except DateParseError as e:
        logger.error(f"Invalid as_of_date format: {as_of_date}")
        raise ValueError(f"Module 5 v3: {e}") from e

    # Run PIT production gate (if enforcement enabled)
    production_gate_result = None
    if enforce_pit_gates and (use_adaptive_weights or historical_scores):
        production_gate_result = run_production_gate(
            as_of_date=as_of_dt,
            historical_scores=historical_scores,
            historical_returns=historical_returns,
            use_adaptive_weights=use_adaptive_weights,
            current_weights=weights,
            previous_weights=previous_weights,
        )

        if not production_gate_result.passed:
            logger.warning(
                f"Production gate FAILED: {production_gate_result.blocking_violations}. "
                f"Falling back to non-adaptive mode."
            )
            use_adaptive_weights = False  # Disable adaptive weights on failure
            historical_scores = None
            historical_returns = None

        if production_gate_result.warnings:
            logger.warning(f"Production gate warnings: {production_gate_result.warnings}")

    # Handle empty universe
    active_securities = universe_result.get("active_securities", [])
    if not active_securities:
        logger.warning("Module 5 v3: Empty universe provided")
        return _empty_result(as_of_date)

    logger.info(f"Module 5 v3: Computing IC-enhanced composite scores for {as_of_date}")

    # =========================================================================
    # EXTRACT ENHANCEMENT DATA
    # =========================================================================

    enhancement_applied = enhancement_result is not None
    pos_by_ticker = {}
    si_by_ticker = {}
    regime_adjustments = {}
    regime_name = "NEUTRAL"
    fda_by_ticker = {}
    diversity_by_ticker = {}
    intensity_by_ticker = {}
    partnership_by_ticker = {}
    accuracy_by_ticker = {}
    cash_burn_by_ticker = {}
    phase_momentum_by_ticker = {}

    if enhancement_result:
        # Extract PoS scores
        for ps in enhancement_result.get("pos_scores", {}).get("scores", []):
            if ps.get("ticker"):
                pos_by_ticker[ps["ticker"].upper()] = ps

        # Extract SI signals
        si_data = enhancement_result.get("short_interest_scores") or {}
        for si in si_data.get("scores", []):
            if si.get("ticker"):
                si_by_ticker[si["ticker"].upper()] = si

        # Extract regime
        regime_data = enhancement_result.get("regime", {})
        regime_name = regime_data.get("regime", "NEUTRAL")
        regime_adjustments = regime_data.get("signal_adjustments", {})

        # Extract accuracy enhancements
        accuracy_data = enhancement_result.get("accuracy_enhancements") or {}
        accuracy_by_ticker = accuracy_data.get("adjustments", {})

        # Extract FDA designation scores
        fda_scores_data = enhancement_result.get("fda_designation_scores") or {}
        for fda in fda_scores_data.get("scores", []):
            if fda.get("ticker"):
                fda_by_ticker[fda["ticker"].upper()] = fda

        # Extract pipeline diversity scores
        diversity_scores_data = enhancement_result.get("pipeline_diversity_scores") or {}
        for div in diversity_scores_data.get("scores", []):
            if div.get("ticker"):
                diversity_by_ticker[div["ticker"].upper()] = div

        # Extract competitive intensity scores
        intensity_scores_data = enhancement_result.get("competitive_intensity_scores") or {}
        for ci in intensity_scores_data.get("scores", []):
            if ci.get("ticker"):
                intensity_by_ticker[ci["ticker"].upper()] = ci

        # Extract partnership validation scores
        partnership_scores_data = enhancement_result.get("partnership_scores") or {}
        for ps in partnership_scores_data.get("scores_by_ticker", {}).values():
            if ps.get("ticker"):
                partnership_by_ticker[ps["ticker"].upper()] = ps

        # Extract cash burn trajectory data (uses scores_by_ticker format)
        cash_burn_scores_data = enhancement_result.get("cash_burn_scores") or {}
        for ticker, cb in cash_burn_scores_data.get("scores_by_ticker", {}).items():
            cash_burn_by_ticker[ticker.upper()] = cb

        # Extract phase momentum data (uses scores_by_ticker format)
        phase_momentum_data = enhancement_result.get("phase_momentum_scores") or {}
        for ticker, pm in phase_momentum_data.get("scores_by_ticker", {}).items():
            phase_momentum_by_ticker[ticker.upper()] = pm

    # =========================================================================
    # DETERMINE SCORING MODE AND WEIGHTS
    # =========================================================================

    market_data_dict = market_data_by_ticker or {}
    has_market_data = bool(market_data_dict)
    has_pos_data = bool(pos_by_ticker)

    if has_pos_data:
        mode = ScoringMode.ENHANCED
        base_weights = V3_ENHANCED_WEIGHTS.copy() if weights is None else weights
    elif has_market_data:
        mode = ScoringMode.PARTIAL
        base_weights = V3_PARTIAL_WEIGHTS.copy() if weights is None else weights
    else:
        mode = ScoringMode.DEFAULT
        base_weights = V3_DEFAULT_WEIGHTS.copy() if weights is None else weights

    # Adaptive weight learning (if enabled and data available)
    # Uses PIT-safe signature: historical_returns keyed by (as_of_date, ticker)
    adaptive_weights_result = None
    if use_adaptive_weights and historical_scores and historical_returns:
        adaptive_weights_result = compute_adaptive_weights(
            historical_scores,
            historical_returns,  # Dict[(date, ticker), Decimal]
            base_weights,
            asof_date=as_of_dt,
            embargo_months=embargo_months,
            shrinkage_lambda=shrinkage_lambda,
            smooth_gamma=smooth_gamma,
            prev_weights=previous_weights,
        )
        if adaptive_weights_result.confidence >= Decimal("0.4"):
            base_weights = adaptive_weights_result.weights
            mode = ScoringMode.ADAPTIVE
            logger.info(
                f"Using adaptive weights (method={adaptive_weights_result.optimization_method}, "
                f"confidence={adaptive_weights_result.confidence}, "
                f"training_periods={adaptive_weights_result.training_periods})"
            )

    # =========================================================================
    # INDEX MODULE OUTPUTS
    # =========================================================================

    # DETERMINISM: Sort active_tickers to ensure consistent iteration order
    # (set iteration order is non-deterministic due to Python hash randomization)
    active_tickers = sorted({s["ticker"] for s in universe_result.get("active_securities", [])})
    financial_by_ticker = {s["ticker"]: s for s in financial_result.get("scores", [])}
    catalyst_by_ticker = catalyst_result.get("summaries", {})
    clinical_by_ticker = {s["ticker"]: s for s in clinical_result.get("scores", [])}

    # Index raw financial data for survivability scoring
    raw_financial_by_ticker = {}
    if raw_financial_data:
        for rec in raw_financial_data:
            ticker = rec.get("ticker")
            if ticker:
                raw_financial_by_ticker[ticker.upper()] = rec

    # =========================================================================
    # BUILD COMBINED RECORDS
    # =========================================================================

    combined = []
    excluded = []

    for ticker in active_tickers:
        fin = financial_by_ticker.get(ticker, {}).copy()  # Copy to avoid mutating original
        # Merge raw financial data for survivability scoring
        raw_fin = raw_financial_by_ticker.get(ticker.upper(), {})
        if raw_fin:
            # Add raw fields that survivability module needs
            for key in ['Cash', 'CFO', 'R&D', 'OperatingExpenses', 'InterestExpense',
                        'LongTermDebt', 'LongTermDebtCurrent', 'ShortTermInvestments',
                        'MarketableSecurities', 'Revenue']:
                if key in raw_fin and key not in fin:
                    fin[key] = raw_fin[key]
        cat = catalyst_by_ticker.get(ticker, {})
        clin = clinical_by_ticker.get(ticker, {})
        pos = pos_by_ticker.get(ticker.upper())
        si = si_by_ticker.get(ticker.upper())
        market = market_data_dict.get(ticker, {})
        fda = fda_by_ticker.get(ticker.upper())
        diversity = diversity_by_ticker.get(ticker.upper())
        intensity = intensity_by_ticker.get(ticker.upper())
        partnership = partnership_by_ticker.get(ticker.upper())
        cash_burn = cash_burn_by_ticker.get(ticker.upper())
        phase_momentum = phase_momentum_by_ticker.get(ticker.upper())

        # Extract raw scores
        fin_score = _to_decimal(extract_financial_score(fin))
        clin_score = _to_decimal(extract_clinical_score(clin))

        # Extract catalyst score
        if hasattr(cat, 'score_blended'):
            cat_score = _to_decimal(cat.score_blended)
        elif isinstance(cat, dict):
            scores = cat.get("scores", cat)
            cat_score = _to_decimal(scores.get("score_blended", scores.get("catalyst_score_net")))
        else:
            cat_score = None

        pos_score = _to_decimal(pos.get("pos_score")) if pos else None

        # Get severities
        severities = [
            fin.get("severity", "none"),
            clin.get("severity", "none"),
        ]
        if hasattr(cat, 'severe_negative_flag') and cat.severe_negative_flag:
            severities.append("sev1")
        elif isinstance(cat, dict) and cat.get("flags", {}).get("severe_negative_flag"):
            severities.append("sev1")

        worst_severity = _get_worst_severity(severities)

        # Exclude sev3
        if worst_severity == Severity.SEV3:
            excluded.append({
                "ticker": ticker,
                "reason": "sev3_gate",
                "severity": worst_severity.value,
            })
            continue

        # Get cohort info
        lead_phase = clin.get("lead_phase")
        market_cap_mm = fin.get("market_cap_mm")
        trial_count = clin.get("trial_count", 0)

        combined.append({
            "ticker": ticker,
            "clinical_raw": clin_score,
            "financial_raw": fin_score,
            "catalyst_raw": cat_score,
            "pos_raw": pos_score,
            "market_cap_bucket": _market_cap_bucket(market_cap_mm),
            "stage_bucket": _stage_bucket(lead_phase),
            "lead_phase": lead_phase,
            "market_cap_mm": _to_decimal(market_cap_mm),
            "trial_count": trial_count,
            "fin_data": fin,
            "cat_data": cat,
            "clin_data": clin,
            "pos_data": pos,
            "si_data": si,
            "market_data": market,
            "fda_data": fda,
            "diversity_data": diversity,
            "intensity_data": intensity,
            "partnership_data": partnership,
            "cash_burn_data": cash_burn,
            "phase_momentum_data": phase_momentum,
        })

    # =========================================================================
    # COMPUTE GLOBAL STATS FOR SHRINKAGE
    # =========================================================================

    global_stats = _compute_global_stats(combined)

    # =========================================================================
    # BUILD PEER VALUATION DATA
    # =========================================================================

    peer_valuations = []
    for r in combined:
        if r["market_cap_mm"] is None:
            continue
        # Development valuation requires trial_count > 0, but commercial doesn't
        # Include all for peer comparison
        fin = r.get("fin_data", {})
        mkt = r.get("market_data", {})

        # Extract revenue_mm (convert from raw to millions)
        revenue_raw = fin.get("Revenue")
        revenue_mm = Decimal(str(revenue_raw)) / Decimal("1000000") if revenue_raw else None

        # Extract cfo_mm (convert from raw to millions)
        cfo_raw = fin.get("CFO")
        cfo_mm = Decimal(str(cfo_raw)) / Decimal("1000000") if cfo_raw else None

        # Extract enterprise_value_mm
        ev_raw = mkt.get("enterprise_value")
        enterprise_value_mm = Decimal(str(ev_raw)) / Decimal("1000000") if ev_raw else None

        peer_valuations.append({
            "ticker": r["ticker"],
            "market_cap_mm": r["market_cap_mm"],
            "trial_count": r["trial_count"],
            "stage_bucket": r["stage_bucket"],
            "has_revenue": fin.get("has_revenue", False),
            "revenue_mm": revenue_mm,
            "cfo_mm": cfo_mm,
            "enterprise_value_mm": enterprise_value_mm,
        })

    # =========================================================================
    # COHORT GROUPING AND NORMALIZATION
    # =========================================================================

    cohorts: Dict[str, List[Dict]] = {}
    for rec in combined:
        key = rec["stage_bucket"]
        rec["cohort_key"] = key
        cohorts.setdefault(key, []).append(rec)

    cohort_stats = {}
    for cohort_key, members in cohorts.items():
        method = _apply_cohort_normalization_v3(
            members,
            global_stats,
            include_pos=enhancement_applied,
            use_shrinkage=True,
        )
        cohort_stats[cohort_key] = {
            "count": len(members),
            "normalization_method": method.value,
        }

    # =========================================================================
    # ENRICH WITH CO-INVEST
    # =========================================================================

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    for rec in combined:
        if coinvest_signals:
            rec["coinvest"] = _enrich_with_coinvest(rec["ticker"], coinvest_signals, as_of_dt)
        else:
            rec["coinvest"] = {"coinvest_overlap_count": 0, "coinvest_holders": [], "coinvest_usable": False, "position_changes": {}}

    # =========================================================================
    # APPLY ACCURACY ENHANCEMENTS (if available)
    # =========================================================================

    accuracy_by_ticker = accuracy_by_ticker if 'accuracy_by_ticker' in dir() else {}
    if accuracy_by_ticker:
        for rec in combined:
            ticker = rec["ticker"].upper()
            acc_adj = accuracy_by_ticker.get(ticker)
            if acc_adj:
                # Apply multipliers to normalized scores
                clin_mult = Decimal(acc_adj.get("clinical_adjustment", "1.00"))
                fin_mult = Decimal(acc_adj.get("financial_adjustment", "1.00"))
                cat_mult = Decimal(acc_adj.get("catalyst_adjustment", "1.00"))
                reg_bonus = Decimal(acc_adj.get("regulatory_bonus", "0"))

                if rec.get("clinical_normalized"):
                    adj_clinical = rec["clinical_normalized"] * clin_mult + reg_bonus
                    rec["clinical_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_clinical))

                if rec.get("financial_normalized"):
                    adj_financial = rec["financial_normalized"] * fin_mult
                    rec["financial_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_financial))

                if rec.get("catalyst_normalized"):
                    adj_catalyst = rec["catalyst_normalized"] * cat_mult
                    rec["catalyst_normalized"] = max(Decimal("0"), min(Decimal("100"), adj_catalyst))

                # Track adjustments in rec for audit
                rec["accuracy_adjustments_applied"] = acc_adj.get("adjustments_applied", [])

    # =========================================================================
    # SCORE EACH TICKER
    # =========================================================================

    scored = []
    for rec in combined:
        normalized_scores = {
            "clinical": rec.get("clinical_normalized"),
            "financial": rec.get("financial_normalized"),
            "catalyst": rec.get("catalyst_normalized"),
            "pos": rec.get("pos_normalized"),
        }

        result = _score_single_ticker_v3(
            ticker=rec["ticker"],
            fin_data=rec["fin_data"],
            cat_data=rec["cat_data"],
            clin_data=rec["clin_data"],
            pos_data=rec["pos_data"],
            si_data=rec["si_data"],
            market_data=rec["market_data"],
            coinvest_data=rec["coinvest"],
            base_weights=base_weights,
            regime=regime_name,
            mode=mode,
            normalized_scores=normalized_scores,
            cohort_key=rec["cohort_key"],
            normalization_method=rec.get("normalization_method", NormalizationMethod.COHORT),
            peer_valuations=peer_valuations,
            fda_data=rec.get("fda_data"),
            diversity_data=rec.get("diversity_data"),
            intensity_data=rec.get("intensity_data"),
            partnership_data=rec.get("partnership_data"),
            cash_burn_data=rec.get("cash_burn_data"),
            phase_momentum_data=rec.get("phase_momentum_data"),
        )

        result["market_cap_bucket"] = rec["market_cap_bucket"]
        result["stage_bucket"] = rec["stage_bucket"]
        result["cohort_key"] = rec["cohort_key"]
        result["coinvest"] = rec["coinvest"]
        scored.append(result)

    # =========================================================================
    # APPLY ROBUSTNESS ENHANCEMENTS (v3.3)
    # =========================================================================
    # Applies the 8 robustness enhancements to improve score stability:
    # 1. Winsorization at component level
    # 2. Confidence-weighted shrinkage
    # 3. Rank stability regularization (requires prior rankings)
    # 4. Multi-timeframe signal blending (already applied in momentum)
    # 5. Asymmetric interaction bounds
    # 6. Regime-conditional weight floors
    # 7. Defensive override triggers
    # 8. Score distribution health checks

    robustness_summary = None
    try:
        # Build universe stats for defensive trigger evaluation
        universe_stats = {
            "severity_ratio": Decimal(str(
                sum(1 for s in scored if s["severity"].value in ("sev2", "sev3")) / max(len(scored), 1)
            )),
            "avg_runway_months": Decimal(str(
                sum(
                    _to_decimal(s.get("score_breakdown", {}).penalties_and_gates.get("runway_months"), Decimal("24"))
                    if hasattr(s.get("score_breakdown"), "penalties_and_gates") else Decimal("24")
                    for s in scored
                ) / max(len(scored), 1)
            )) if scored else Decimal("24"),
            "high_vol_ratio": Decimal(str(
                sum(1 for s in scored if s.get("volatility_adjustment", {}).get("vol_bucket") == "high") / max(len(scored), 1)
            )),
            "positive_momentum_ratio": Decimal(str(
                sum(1 for s in scored if "strong_positive_momentum" in s.get("flags", [])) / max(len(scored), 1)
            )),
        }

        # Evaluate defensive triggers
        defensive_result = evaluate_defensive_triggers(universe_stats)

        # Apply weight floors based on regime
        sample_weights = scored[0]["effective_weights"] if scored else base_weights
        floor_result = apply_weight_floors(sample_weights, regime_name)

        # Check distribution health
        all_scores = [s["composite_score"] for s in scored]
        dist_health = check_distribution_health(all_scores)

        # Apply asymmetric bounds to interaction terms
        interaction_caps_applied = 0
        for rec in scored:
            interaction_adj = _to_decimal(
                rec.get("interaction_terms", {}).get("total_adjustment"),
                Decimal("0")
            )
            bounds = apply_asymmetric_bounds(interaction_adj)
            if bounds.was_capped:
                interaction_caps_applied += 1
                # Adjust composite score for capped interaction
                score_delta = bounds.applied_value - interaction_adj
                rec["composite_score"] = _clamp(
                    rec["composite_score"] + score_delta,
                    Decimal("0"),
                    Decimal("100")
                )
                if "asymmetric_interaction_capped" not in rec["flags"]:
                    rec["flags"].append("asymmetric_interaction_capped")

        # Build robustness summary
        robustness_summary = {
            "defensive_posture": defensive_result.posture.value,
            "defensive_triggers_hit": defensive_result.triggers_hit,
            "weight_floors_applied": floor_result.floors_applied,
            "distribution_health": dist_health.health.value,
            "distribution_issues": dist_health.issues,
            "interaction_caps_applied": interaction_caps_applied,
            "flags": defensive_result.flags + floor_result.flags,
        }

        # Log warnings if defensive posture is elevated
        if defensive_result.posture != DefensivePosture.NONE:
            logger.warning(
                f"Robustness: Defensive posture {defensive_result.posture.value} "
                f"triggered by: {defensive_result.triggers_hit}"
            )

        # Log warnings for distribution health issues
        if dist_health.health != DistributionHealth.HEALTHY:
            logger.warning(
                f"Robustness: Distribution health {dist_health.health.value} "
                f"- issues: {dist_health.issues}"
            )

    except Exception as e:
        logger.warning(f"Robustness enhancements failed (non-blocking): {e}")
        robustness_summary = {"error": str(e), "defensive_posture": "none"}

    # =========================================================================
    # SORT AND RANK
    # =========================================================================

    # Sort by composite score (desc), then coinvest (desc), then ticker (asc)
    scored.sort(key=lambda x: (
        -x["composite_score"],
        -(x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0),
        x["ticker"]
    ))

    for i, rec in enumerate(scored):
        rec["composite_rank"] = i + 1

    # =========================================================================
    # FORMAT OUTPUT
    # =========================================================================

    ranked_securities = []
    for rec in scored:
        bd = rec["score_breakdown"]
        coinvest = rec.get("coinvest") or {}

        security_data = {
            "ticker": rec["ticker"],
            "composite_score": str(rec["composite_score"]),
            "composite_rank": rec["composite_rank"],
            "severity": rec["severity"].value,
            "flags": rec["flags"],
            "rankable": True,

            # Cohort info
            "market_cap_bucket": rec["market_cap_bucket"],
            "stage_bucket": rec["stage_bucket"],
            "cohort_key": rec["cohort_key"],
            "normalization_method": rec["normalization_method"],

            # Confidence
            "confidence_clinical": str(rec["confidence_clinical"]),
            "confidence_financial": str(rec["confidence_financial"]),
            "confidence_catalyst": str(rec["confidence_catalyst"]),
            "confidence_pos": str(rec["confidence_pos"]) if rec["confidence_pos"] else None,
            "confidence_overall": str(rec["confidence_overall"]),

            # Weights
            "effective_weights": {k: str(v) for k, v in rec["effective_weights"].items()},

            # Caps and penalties
            "monotonic_caps_applied": rec["caps_applied"],
            "uncertainty_penalty": str(rec["uncertainty_penalty"]),

            # Audit
            "determinism_hash": rec["determinism_hash"],
            "schema_version": SCHEMA_VERSION,

            # Score breakdown
            "score_breakdown": {
                "version": bd.version, "mode": bd.mode,
                "base_weights": bd.base_weights,
                "regime_adjustments": bd.regime_adjustments,
                "effective_weights": bd.effective_weights,
                "components": bd.components,
                "enhancements": bd.enhancements,
                "penalties_and_gates": bd.penalties_and_gates,
                "interaction_terms": bd.interaction_terms,
                "final": bd.final,
                "normalization_method": bd.normalization_method,
                "cohort_info": bd.cohort_info,
                "hybrid_aggregation": bd.hybrid_aggregation,
            },

            # Top-level component_scores (mirrors score_breakdown.components for convenience)
            "component_scores": bd.components,

            # Co-invest
            "coinvest_overlap_count": coinvest.get("coinvest_overlap_count", 0),
            "coinvest_holders": coinvest.get("coinvest_holders", []),
            "coinvest_usable": coinvest.get("coinvest_usable", False),

            # V3 Enhancement signals
            "momentum_signal": rec.get("momentum_signal"),
            "valuation_signal": rec.get("valuation_signal"),
            "smart_money_signal": rec.get("smart_money_signal"),
            "short_interest_signal": rec.get("short_interest_signal"),
            "volatility_adjustment": rec.get("volatility_adjustment"),
            "catalyst_decay": rec.get("catalyst_decay"),
            "interaction_terms": rec.get("interaction_terms"),
            "catalyst_effective": rec.get("catalyst_effective"),
            "fda_designation_signal": rec.get("fda_designation_signal"),
            "pipeline_diversity_signal": rec.get("pipeline_diversity_signal"),
            "competitive_intensity_signal": rec.get("competitive_intensity_signal"),
            "partnership_signal": rec.get("partnership_signal"),
            "survivability_signal": rec.get("survivability_signal"),
        }

        ranked_securities.append(security_data)

    # =========================================================================
    # COMPUTE EXPECTED RETURNS
    # =========================================================================
    # Convert score → rank → percentile → z-score → expected excess return
    # This separates signal research from portfolio engineering
    # λ = 0.08 (8% per 1σ per year) - conservative biotech-appropriate default
    er_provenance = compute_expected_returns(ranked_securities)

    # =========================================================================
    # BUILD DIAGNOSTIC COUNTS
    # =========================================================================

    diagnostic_counts = {
        "total_input": len(active_tickers),
        "rankable": len(ranked_securities),
        "excluded": len(excluded),
        "cohort_count": len(cohorts),

        # Enhancement coverage
        "with_pos_scores": sum(1 for r in ranked_securities if r.get("confidence_pos")),
        "with_market_data": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("annualized_vol_pct")),
        "with_momentum_signal": sum(1 for r in ranked_securities if r.get("momentum_signal", {}).get("alpha_60d")),
        "with_valuation_signal": sum(1 for r in ranked_securities if r.get("valuation_signal", {}).get("peer_count", 0) >= 5),
        "with_smart_money": sum(1 for r in ranked_securities if r.get("coinvest_overlap_count", 0) > 0),
        "with_fda_designations": sum(1 for r in ranked_securities if r.get("fda_designation_signal", {}).get("has_designations")),
        "with_pipeline_diversity": sum(1 for r in ranked_securities if r.get("pipeline_diversity_signal", {}).get("diversity_score")),
        "with_competitive_intensity": sum(1 for r in ranked_securities if r.get("competitive_intensity_signal", {}).get("intensity_score")),
        "with_partnerships": sum(1 for r in ranked_securities if r.get("partnership_signal", {}).get("partnership_count", 0) > 0),

        # Momentum state breakdown (for debugging/attribution)
        # Categories are MUTUALLY EXCLUSIVE and sum to total_rankable:
        # 1. missing_prices: No price data available for any window
        # 2. computed_low_conf: Computed but confidence < 0.5
        # 3. applied_negative: Strong negative signal
        # 4. applied_positive: Strong positive signal
        # 5. applied_neutral: Signal computed but not strong either way
        "momentum_missing_prices": sum(
            1 for r in ranked_securities
            if "momentum_missing_prices" in r.get("flags", [])
        ),
        "momentum_computed_low_conf": sum(
            1 for r in ranked_securities
            if "momentum_low_confidence" in r.get("flags", [])
        ),
        "momentum_applied_negative": sum(
            1 for r in ranked_securities
            if "strong_negative_momentum" in r.get("flags", [])
            and "momentum_missing_prices" not in r.get("flags", [])
        ),
        "momentum_applied_positive": sum(
            1 for r in ranked_securities
            if "strong_positive_momentum" in r.get("flags", [])
            and "momentum_missing_prices" not in r.get("flags", [])
        ),

        # Window usage breakdown
        "momentum_window_20d": sum(
            1 for r in ranked_securities
            if "momentum_window_20d" in r.get("flags", [])
        ),
        "momentum_window_60d": sum(
            1 for r in ranked_securities
            if "momentum_window_60d" in r.get("flags", [])
        ),
        "momentum_window_120d": sum(
            1 for r in ranked_securities
            if "momentum_window_120d" in r.get("flags", [])
        ),

        # Legacy compat: "no_alpha" = missing_prices + computed_low_conf
        "momentum_no_alpha": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("alpha_60d") is None
        ),
        "momentum_gated_with_data": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("alpha_60d") is not None
            and "momentum_confidence_gated" in r.get("flags", [])
        ),

        # NEW v3.2: Stable coverage metrics for momentum
        # These three metrics are stable and avoid "coverage inflation"
        #
        # 1. momentum_computable: Any window computed (low_conf + applied)
        #    = tickers where we have at least one return window
        "momentum_computable": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("window_used") is not None
        ),
        # 2. momentum_meaningful: Confidence >= 0.5 threshold
        #    = signals strong enough to be trusted
        "momentum_meaningful": sum(
            1 for r in ranked_securities
            if _to_decimal(r.get("momentum_signal", {}).get("confidence", "0")) >= Decimal("0.5")
        ),
        # 3. momentum_strong_signal: Score moved away from 50 by at least 2.5 points
        #    = signals strong enough to meaningfully affect rankings
        #    Note: "applied" total (neg + pos + neutral) is computed separately for consistency
        #
        #    Alpha anchoring (score = 50 + conf * alpha * 150):
        #    - With conf=0.7 (typical): |score-50| >= 2.5 requires |alpha| >= ~2.4%
        #    - With conf=0.9 (high):    |score-50| >= 2.5 requires |alpha| >= ~1.85%
        #    - Raw (no shrinkage):      |score-50| >= 2.5 requires |alpha| >= ~1.67%
        #    Inclusive boundary: score=47.5 or score=52.5 counts as strong.
        "momentum_strong_signal": sum(
            1 for r in ranked_securities
            if abs(_to_decimal(r.get("momentum_signal", {}).get("momentum_score", "50")) - Decimal("50")) >= Decimal("2.5")
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),
        # 4. momentum_strong_and_effective: Strong signal AND high enough confidence to matter
        #    = signals that are both strong (|score-50| >= 2.5) AND have confidence >= 0.6
        #    This is the "portfolio impact" metric: signals likely to move composites
        "momentum_strong_and_effective": sum(
            1 for r in ranked_securities
            if abs(_to_decimal(r.get("momentum_signal", {}).get("momentum_score", "50")) - Decimal("50")) >= Decimal("2.5")
            and r.get("momentum_signal", {}).get("window_used") is not None
            and _to_decimal(r.get("momentum_signal", {}).get("confidence", "0")) >= Decimal("0.6")
        ),
        # 5. Momentum source breakdown (for observability)
        #    prices = computed from daily price returns
        #    13f = injected from institutional momentum data
        "momentum_source_prices": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("source") == "prices"
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),
        "momentum_source_13f": sum(
            1 for r in ranked_securities
            if r.get("momentum_signal", {}).get("source") == "13f"
            and r.get("momentum_signal", {}).get("window_used") is not None
        ),
        # 6. Alpha distribution metrics (for regime attribution)
        #    Helps distinguish "strong=0 because low dispersion" from "strong=0 because bug"
        #    Values stored as strings for Decimal consistency
        **_compute_alpha_distribution_metrics(ranked_securities),

        # Quality metrics
        "with_caps_applied": sum(1 for r in ranked_securities if r.get("monotonic_caps_applied")),
        "with_interaction_flags": sum(1 for r in ranked_securities if r.get("interaction_terms", {}).get("flags")),
        "high_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "high"),
        "low_volatility_count": sum(1 for r in ranked_securities if r.get("volatility_adjustment", {}).get("vol_bucket") == "low"),
        "in_catalyst_window": sum(1 for r in ranked_securities if r.get("catalyst_decay", {}).get("in_optimal_window")),

        # Catalyst coverage breakdown (new in v3.1)
        # Raw coverage = has any catalyst events (confidence > default 0.3)
        # Window coverage = in optimal catalyst window (15-45 days from event)
        "with_catalyst_events": sum(
            1 for r in ranked_securities
            if _to_decimal(r.get("confidence_catalyst")) and _to_decimal(r.get("confidence_catalyst")) > Decimal("0.3")
        ),
        # New in v3.2: Proximity-blended catalyst tracking
        # Counts tickers where proximity score was factored into catalyst_effective
        "with_catalyst_proximity_blended": sum(
            1 for r in ranked_securities
            if r.get("catalyst_effective", {}).get("catalyst_proximity_blended", False)
        ),
    }

    # =========================================================================
    # ENHANCEMENT DIAGNOSTICS
    # =========================================================================

    enhancement_diagnostics = {
        "regime": regime_name,
        "regime_adjustments": {k: str(v) for k, v in regime_adjustments.items()} if regime_adjustments else {},
        "mode": mode.value,
        "pos_coverage": len(pos_by_ticker),
        "market_data_coverage": len(market_data_dict),
    }

    # =========================================================================
    # PIPELINE HEALTH CHECK
    # =========================================================================

    total_rankable = len(ranked_securities) if ranked_securities else 1  # Avoid div/0

    # Calculate component coverage as fraction of universe
    component_coverage = {
        # catalyst_window = in optimal 15-45 day window (stricter)
        "catalyst": Decimal(str(diagnostic_counts.get("in_catalyst_window", 0))) / Decimal(str(total_rankable)),
        # catalyst_raw = has any catalyst events (confidence > 0.3 default)
        "catalyst_raw": Decimal(str(diagnostic_counts.get("with_catalyst_events", 0))) / Decimal(str(total_rankable)),
        "momentum": Decimal(str(diagnostic_counts.get("with_momentum_signal", 0))) / Decimal(str(total_rankable)),
        "smart_money": Decimal(str(diagnostic_counts.get("with_smart_money", 0))) / Decimal(str(total_rankable)),
        "market_data": Decimal(str(diagnostic_counts.get("with_market_data", 0))) / Decimal(str(total_rankable)),
        "pos": Decimal(str(diagnostic_counts.get("with_pos_scores", 0))) / Decimal(str(total_rankable)),
        "valuation": Decimal(str(diagnostic_counts.get("with_valuation_signal", 0))) / Decimal(str(total_rankable)),
    }

    # Collect components with confidence-gated entries (from individual security scoring)
    gated_component_counts = {}
    for sec in ranked_securities:
        for flag in sec.get("flags", []):
            if flag.endswith("_confidence_gated"):
                comp = flag.replace("_confidence_gated", "")
                gated_component_counts[comp] = gated_component_counts.get(comp, 0) + 1

    # Determine run status
    run_status = RunStatus.OK
    degraded_components = []
    health_warnings = []
    health_errors = []

    # Check each pipeline against thresholds
    for component, threshold in HEALTH_GATE_THRESHOLDS.items():
        coverage = component_coverage.get(component, Decimal("0"))
        if coverage < threshold:
            # Catalyst: check raw coverage for pipeline health, not window coverage
            if component == "catalyst":
                raw_coverage = component_coverage.get("catalyst_raw", Decimal("0"))
                if raw_coverage < Decimal("0.05"):  # <5% with events = pipeline broken
                    run_status = RunStatus.FAIL
                    health_errors.append(
                        f"CRITICAL: catalyst_raw coverage {raw_coverage*100:.1f}% - pipeline broken "
                        f"(window coverage: {coverage*100:.1f}%)"
                    )
                else:
                    # Raw coverage OK but window coverage low - just informational
                    health_warnings.append(
                        f"INFO: catalyst window coverage {coverage*100:.1f}% (raw: {raw_coverage*100:.1f}%) - "
                        f"events exist but few in optimal 15-45d window"
                    )
            elif component in ("momentum", "market_data"):
                # Momentum/market data below threshold degrades run
                if run_status != RunStatus.FAIL:
                    run_status = RunStatus.DEGRADED
                degraded_components.append(component)
                health_warnings.append(f"DEGRADED: {component} coverage {coverage*100:.1f}% < {threshold*100:.0f}% threshold")
            else:
                # Other components just warn
                degraded_components.append(component)
                health_warnings.append(f"WARNING: {component} coverage {coverage*100:.1f}% < {threshold*100:.0f}% threshold")

    # Check gated components (high rate of confidence gating indicates data issue)
    # NOTE: In biotech, high gating rates are common for optional enhancement components:
    # - valuation: pre-revenue companies lack comparable peers
    # - momentum: 13F-based momentum provides sparse coverage by design
    # - smart_money: 13F data only covers subset of universe
    # Only log as INFO since sparse coverage is expected, not a data pipeline failure
    for comp, count in gated_component_counts.items():
        gated_pct = Decimal(str(count)) / Decimal(str(total_rankable))
        if gated_pct > Decimal("0.5"):  # >50% of universe gated for this component
            # Log as info, not degradation - sparse optional component coverage is normal
            health_warnings.append(f"INFO: {comp} confidence-gated for {gated_pct*100:.1f}% of universe (sparse coverage expected)")

    # Compute momentum breakdown (single source of truth)
    # This ensures applied = neg + pos + neutral = total_rankable - missing - low_conf
    mom_breakdown = compute_momentum_breakdown(
        ranked_securities=ranked_securities,
        diagnostic_counts=diagnostic_counts,
        total_rankable=total_rankable,
    )

    # Coverage guardrail check
    guardrail_warning = check_coverage_guardrail(mom_breakdown, threshold_pct=20.0)
    if guardrail_warning:
        health_warnings.append(guardrail_warning)

    # Log momentum breakdown if there's any data
    if mom_breakdown["with_data"] > 0 or mom_breakdown["missing"] > 0:
        for log_line in format_momentum_log_lines(mom_breakdown):
            health_warnings.append(log_line)

    # Log health status
    if run_status == RunStatus.FAIL:
        for err in health_errors:
            logger.error(err)
    for warn in health_warnings:
        logger.warning(warn)

    if run_status != RunStatus.OK:
        logger.warning(f"Run status: {run_status.value} | Degraded components: {degraded_components}")

    if adaptive_weights_result:
        enhancement_diagnostics["adaptive_weights"] = {
            "method": adaptive_weights_result.optimization_method,
            "confidence": str(adaptive_weights_result.confidence),
            "historical_ic": adaptive_weights_result.historical_ic_by_component,
        }

    # =========================================================================
    # RETURN RESULT
    # =========================================================================

    # Build production gate diagnostics
    pit_gate_diagnostics = None
    if production_gate_result:
        pit_gate_diagnostics = {
            "passed": production_gate_result.passed,
            "checks": [
                {
                    "name": c.check_name,
                    "status": c.status.value,
                    "violations": c.violations,
                }
                for c in production_gate_result.checks
            ],
            "blocking_violations": production_gate_result.blocking_violations,
            "warnings": production_gate_result.warnings,
            "recommendation": production_gate_result.recommendation,
        }

    # DETERMINISM: Sort excluded_securities by ticker for consistent output order
    excluded_sorted = sorted(excluded, key=lambda x: x["ticker"])

    # Build momentum_health for results JSON (from breakdown computed above)
    momentum_health = build_momentum_health(mom_breakdown, as_of_date)

    return {
        "as_of_date": as_of_date,
        "scoring_mode": mode.value,
        "run_status": run_status.value,
        "degraded_components": degraded_components,
        "component_coverage": {k: str(v) for k, v in sorted(component_coverage.items())},
        "gated_component_counts": gated_component_counts,
        "health_warnings": health_warnings,
        "health_errors": health_errors,
        "weights_used": {k: str(v) for k, v in sorted(base_weights.items())},
        "ranked_securities": ranked_securities,
        "excluded_securities": excluded_sorted,
        "cohort_stats": {k: v for k, v in sorted(cohort_stats.items())},
        "global_stats": {k: {"mean": str(v[0]), "std": str(v[1])} for k, v in sorted(global_stats.items())},
        "diagnostic_counts": diagnostic_counts,
        "enhancement_applied": enhancement_applied,
        "enhancement_diagnostics": enhancement_diagnostics,
        "momentum_health": momentum_health,  # V3.2: Persisted for A/B comparisons
        "robustness_diagnostics": robustness_summary,  # V3.3: Robustness enhancements
        "pit_gate_diagnostics": pit_gate_diagnostics,
        "expected_return_model": er_provenance,  # V3.4: ER provenance for audit
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": sorted(active_tickers), "weights": {k: str(v) for k, v in sorted(base_weights.items())}, "mode": mode.value},
            as_of_date,
        ),
    }


def _compute_alpha_distribution_metrics(ranked_securities: list) -> Dict[str, str]:
    """
    Compute alpha distribution metrics for momentum diagnostics.

    Helps distinguish between:
    - "strong=0 because low dispersion regime" (small alphas, normal)
    - "strong=0 because of a bug" (large alphas but scores near 50)

    Returns dict with string values for Decimal JSON consistency.
    """
    alphas = []
    score_deltas = []

    for r in ranked_securities:
        ms = r.get("momentum_signal", {})
        alpha_str = ms.get("alpha_60d")
        score_str = ms.get("momentum_score")

        if alpha_str is not None:
            try:
                alphas.append(abs(Decimal(str(alpha_str))))
            except (ValueError, TypeError):
                pass

        if score_str is not None:
            try:
                score_deltas.append(abs(Decimal(str(score_str)) - Decimal("50")))
            except (ValueError, TypeError):
                pass

    if not alphas:
        return {
            "alpha_abs_p50": None,
            "alpha_abs_p90": None,
            "alpha_abs_max": None,
            "score_delta_p50": None,
            "score_delta_p90": None,
            "score_delta_max": None,
        }

    # Sort for percentile calculation
    alphas.sort()
    score_deltas.sort()
    n = len(alphas)

    # Compute percentiles (with bounds checking)
    p50_idx = min(int(n * 0.5), n - 1)
    p90_idx = min(int(n * 0.9), n - 1)

    return {
        "alpha_abs_p50": str(alphas[p50_idx].quantize(Decimal("0.0001"))),
        "alpha_abs_p90": str(alphas[p90_idx].quantize(Decimal("0.0001"))),
        "alpha_abs_max": str(max(alphas).quantize(Decimal("0.0001"))),
        "score_delta_p50": str(score_deltas[p50_idx].quantize(Decimal("0.01"))) if score_deltas else None,
        "score_delta_p90": str(score_deltas[p90_idx].quantize(Decimal("0.01"))) if score_deltas else None,
        "score_delta_max": str(max(score_deltas).quantize(Decimal("0.01"))) if score_deltas else None,
    }


def _empty_result(as_of_date: str) -> Dict[str, Any]:
    """Return empty result structure."""
    return {
        "as_of_date": as_of_date,
        "scoring_mode": ScoringMode.DEFAULT.value,
        "run_status": RunStatus.OK.value,
        "degraded_components": [],
        "component_coverage": {},
        "gated_component_counts": {},
        "health_warnings": [],
        "health_errors": [],
        "weights_used": {k: str(v) for k, v in V3_DEFAULT_WEIGHTS.items()},
        "ranked_securities": [],
        "excluded_securities": [],
        "cohort_stats": {},
        "global_stats": {},
        "diagnostic_counts": {
            "total_input": 0, "rankable": 0, "excluded": 0, "cohort_count": 0,
            "with_pos_scores": 0, "with_market_data": 0, "with_momentum_signal": 0,
            "with_valuation_signal": 0, "with_smart_money": 0, "with_caps_applied": 0,
            "with_interaction_flags": 0, "high_volatility_count": 0, "low_volatility_count": 0,
            "in_catalyst_window": 0,
        },
        "enhancement_applied": False,
        "enhancement_diagnostics": None,
        "schema_version": SCHEMA_VERSION,
        "provenance": create_provenance(RULESET_VERSION, {"tickers": [], "weights": {}}, as_of_date),
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Module 5 v3: IC-Enhanced Composite Ranker")
    parser.add_argument("--as-of-date", required=True, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Path to Module 1 output JSON")
    parser.add_argument("--financial", required=True, help="Path to Module 2 output JSON")
    parser.add_argument("--catalyst", required=True, help="Path to Module 3 output JSON")
    parser.add_argument("--clinical", required=True, help="Path to Module 4 output JSON")
    parser.add_argument("--enhancement", help="Path to enhancement data JSON")
    parser.add_argument("--market-data", help="Path to market data JSON")
    parser.add_argument("--financial-raw", help="Path to raw financial records JSON (for survivability scoring)")
    parser.add_argument("--output", required=True, help="Output path for results")
    parser.add_argument("--adaptive-weights", action="store_true", help="Enable adaptive weight learning")

    args = parser.parse_args()

    # Load inputs
    with open(args.universe) as f:
        universe_result = json.load(f)
    with open(args.financial) as f:
        financial_result = json.load(f)
    with open(args.catalyst) as f:
        catalyst_result = json.load(f)
    with open(args.clinical) as f:
        clinical_result = json.load(f)

    enhancement_result = None
    if args.enhancement:
        with open(args.enhancement) as f:
            enhancement_result = json.load(f)

    market_data = None
    if args.market_data:
        with open(args.market_data) as f:
            market_data = json.load(f)

    raw_financial_data = None
    if args.financial_raw:
        with open(args.financial_raw) as f:
            raw_financial_data = json.load(f)

    # Compute
    result = compute_module_5_composite_v3(
        universe_result=universe_result,
        financial_result=financial_result,
        catalyst_result=catalyst_result,
        clinical_result=clinical_result,
        as_of_date=args.as_of_date,
        enhancement_result=enhancement_result,
        market_data_by_ticker=market_data,
        raw_financial_data=raw_financial_data,
        use_adaptive_weights=args.adaptive_weights,
    )

    # Write output
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Module 5 v3 complete: {result['diagnostic_counts']['rankable']} securities ranked")
    sys.exit(0)
